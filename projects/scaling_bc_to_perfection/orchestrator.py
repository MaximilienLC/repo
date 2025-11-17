"""
GUI Orchestrator for managing parallel experiment runs.
Supports Local, SSH Remote, and SLURM Cluster execution backends.
"""

import ctypes
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod
import queue
import time
import json
import os

# Try to import paramiko for SSH support
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None  # type: ignore


class TaskStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    KILLED = "Killed"


class BackendType(Enum):
    LOCAL = "Local"
    SSH = "SSH Remote"
    SLURM = "SLURM Cluster"


@dataclass
class SSHConfig:
    host: str = ""
    port: int = 22
    username: str = ""
    key_file: str = ""
    password: str = ""  # Alternative to key_file
    proxy_jump: str = ""  # e.g., "user@jumphost.com" for ProxyJump


@dataclass
class SLURMConfig:
    ssh_config: SSHConfig = field(default_factory=SSHConfig)
    partition: str = "default"
    time_limit: str = "1:00:00"
    extra_flags: str = ""  # e.g., "--gpus=1 --ntasks=1"


@dataclass
class Task:
    id: int
    command: str
    status: TaskStatus
    backend_type: BackendType = BackendType.LOCAL
    ssh_config: Optional[SSHConfig] = None
    slurm_config: Optional[SLURMConfig] = None
    remote_workdir: str = ""  # Working directory on remote machine (for SSH/SLURM)
    backend: Optional[Any] = None  # ExecutionBackend instance
    output_buffer: list[str] = field(default_factory=list)
    last_poll_time: float = 0.0
    tmux_session: Optional[str] = None


# ============================================================================
# Execution Backends
# ============================================================================


class ExecutionBackend(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    def start(self, command: str, working_dir: str) -> None:
        """Start executing the command."""
        pass

    @abstractmethod
    def poll_output(self) -> list[str]:
        """Get new output lines since last poll."""
        pass

    @abstractmethod
    def get_status(self) -> TaskStatus:
        """Get current task status."""
        pass

    @abstractmethod
    def kill(self) -> None:
        """Kill the running task."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class LocalBackend(ExecutionBackend):
    """Backend for local execution using tmux for resilience."""

    _tmux_configured = False  # Class-level flag for one-time config
    # Full paths to itmux binaries
    _tmux_path = r"C:\Users\Max\Documents\itmux\bin\tmux.exe"
    _bash_path = r"C:\Users\Max\Documents\itmux\bin\bash.exe"

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        self._status = TaskStatus.PENDING
        self._output_buffer: list[str] = []
        self.tmux_session = f"orch_local_task_{task_id}"
        self._last_output_length = 0
        self._script_path: Optional[str] = None  # Temp script file path
        self._log_path: Optional[str] = None  # Output log file path

    def _run_command(self, command: list[str]) -> tuple[int, str, str]:
        """Run a local command and return exit code, stdout, and stderr."""
        # Replace 'tmux' with full path
        if command and command[0] == "tmux":
            command = [self._tmux_path] + command[1:]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        return process.returncode, process.stdout, process.stderr

    def _to_unix_path(self, windows_path: str) -> str:
        """Convert Windows path to Unix format for bash (e.g., C:\\foo -> /c/foo)."""
        path = Path(windows_path).resolve()
        # Convert to POSIX-style first
        posix = path.as_posix()
        # Check if it starts with a drive letter (e.g., C:/)
        if len(posix) >= 2 and posix[1] == ":":
            # Convert C:/path to /c/path (MSYS2/Git Bash format)
            drive_letter = posix[0].lower()
            return f"/{drive_letter}{posix[2:]}"
        return posix

    def _ensure_tmux_configured(self) -> None:
        """Ensure tmux is configured with remain-on-exit globally (one-time setup)."""
        if not LocalBackend._tmux_configured:
            # Start a dummy server to set global option, or set it if server exists
            self._run_command(["tmux", "set-option", "-g", "remain-on-exit", "on"])
            LocalBackend._tmux_configured = True

    def start(self, command: str, working_dir: str) -> None:
        """Execute command locally inside a tmux session."""
        self._status = TaskStatus.RUNNING

        try:
            # Ensure tmux is configured properly
            self._ensure_tmux_configured()

            # Kill any existing session
            self._run_command(["tmux", "kill-session", "-t", self.tmux_session])
            time.sleep(0.1)

            # Create tmux session and run command
            # Use Windows path with forward slashes (itmux bash understands these)
            win_workdir = working_dir.replace("\\", "/")

            # Create a temporary bash script to avoid tmux command parsing issues
            import tempfile

            # Create log file for output
            log_fd, log_path = tempfile.mkstemp(suffix='.log', prefix='orch_output_')
            os.close(log_fd)
            self._log_path = log_path
            win_log_path = log_path.replace("\\", "/")

            # Use Unix line endings (\n) for bash script
            # Redirect all output to log file so we can read it reliably
            script_lines = [
                '#!/bin/bash',
                f'exec > "{win_log_path}" 2>&1',  # Redirect stdout and stderr to log file
                'echo "--- Start of Task (Local) ---"',
                f'echo "Intended Dir: {win_workdir}"',
                f'echo "Command: {command}"',
                'echo "--- Output ---"',
                f'cd "{win_workdir}" && {command}',
                'exit_code=$?',
                'echo "[TASK_EXIT_CODE:$exit_code]"',
                'exit $exit_code',
            ]
            script_content = '\n'.join(script_lines) + '\n'

            # Write script to temp file with Unix line endings
            script_fd, script_path = tempfile.mkstemp(suffix='.sh', prefix='orch_task_')
            try:
                os.write(script_fd, script_content.encode('utf-8'))
                os.close(script_fd)

                # Use Windows paths with forward slashes (itmux bash understands these)
                win_script_path = script_path.replace("\\", "/")
                bash_win_path = self._bash_path.replace("\\", "/")

                # Create session that runs the script
                full_command = f'"{bash_win_path}" "{win_script_path}"'

                # Create session with remain-on-exit so we can capture output after command finishes
                tmux_cmd = [
                    "tmux", "new-session", "-d", "-s", self.tmux_session,
                    "-x", "200", "-y", "50",  # Set window size for better output capture
                    full_command
                ]

                ret_code, stdout, stderr = self._run_command(tmux_cmd)
                print(f"[DEBUG] tmux new-session ret_code={ret_code}, stdout={stdout.strip()}, stderr={stderr.strip()}")

                if ret_code != 0:
                    # Double-check if session actually exists despite error
                    time.sleep(0.2)  # Give tmux time to create session
                    check_ret, check_out, check_err = self._run_command(["tmux", "has-session", "-t", self.tmux_session])
                    print(f"[DEBUG] has-session check: ret={check_ret}, out={check_out.strip()}, err={check_err.strip()}")
                    if check_ret == 0:
                        # Session exists, ignore the error
                        print(f"[DEBUG] Session {self.tmux_session} exists despite ret_code={ret_code}, proceeding")
                        self._output_buffer.append(f"[Started tmux session: {self.tmux_session}]")
                        self._script_path = script_path
                    else:
                        self._status = TaskStatus.FAILED
                        self._output_buffer.append(f"[tmux creation failed: ret={ret_code}, err={stderr}]")
                        # Clean up script file on failure
                        try:
                            os.unlink(script_path)
                        except Exception:
                            pass
                        try:
                            os.unlink(log_path)
                        except Exception:
                            pass
                else:
                    self._output_buffer.append(f"[Started tmux session: {self.tmux_session}]")
                    # Store script path for cleanup later
                    self._script_path = script_path
            except Exception as e:
                os.close(script_fd)
                os.unlink(script_path)
                raise e

        except FileNotFoundError:
            self._status = TaskStatus.FAILED
            self._output_buffer.append("Error: tmux is not installed or not in PATH.")
        except Exception as e:
            self._status = TaskStatus.FAILED
            self._output_buffer.append(f"Local Error: {str(e)}")

    def poll_output(self) -> list[str]:
        """Get new output from the log file."""
        new_lines: list[str] = []

        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return new_lines

        if not self._log_path:
            return new_lines

        try:
            # Read from log file instead of tmux capture-pane
            with open(self._log_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.read().splitlines()

            # Debug: show first poll output
            if self._last_output_length == 0 and all_lines:
                print(f"[DEBUG] First log read for {self.tmux_session} ({len(all_lines)} lines total):")
                # Show all non-empty lines
                non_empty = [(i, line) for i, line in enumerate(all_lines) if line.strip()]
                for i, line in non_empty[:10]:  # Show first 10 non-empty lines
                    print(f"  {i}: {line}")
                if len(non_empty) > 10:
                    print(f"  ... and {len(non_empty) - 10} more non-empty lines")

            if len(all_lines) > self._last_output_length:
                new_lines = all_lines[self._last_output_length :]
                # Debug: show when new lines are captured
                non_empty_new = [l for l in new_lines if l.strip()]
                if non_empty_new and self._last_output_length > 0:
                    print(f"[DEBUG] {self.tmux_session}: {len(new_lines)} new lines")
                self._output_buffer.extend(new_lines)
                self._last_output_length = len(all_lines)

                # Check for exit code marker
                for line in new_lines:
                    if "[TASK_EXIT_CODE:" in line:
                        try:
                            code = int(line.split(":")[1].rstrip("]"))
                            self._status = TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
                            self._output_buffer.append(f"[Exit code: {code}]")
                        except (ValueError, IndexError):
                            pass

        except FileNotFoundError:
            pass  # Log file not created yet
        except Exception as e:
            new_lines.append(f"[Poll Error: {str(e)}]")

        return new_lines

    def get_status(self) -> TaskStatus:
        """Check if the local tmux session is still running."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return self._status

        try:
            ret_code, stdout, stderr = self._run_command(["tmux", "has-session", "-t", self.tmux_session])
            print(f"[DEBUG] has-session for {self.tmux_session}: ret={ret_code}, stderr={stderr.strip()}")
            if ret_code != 0:
                # Session is gone. If we were running, mark as completed.
                if self._status == TaskStatus.RUNNING:
                    self._status = TaskStatus.COMPLETED
                    self._output_buffer.append("[tmux session ended]")
        except Exception as e:
            print(f"[DEBUG] has-session exception: {e}")
            pass  # Keep current status on error

        return self._status

    def kill(self) -> None:
        """Kill the local tmux session."""
        self._status = TaskStatus.KILLED
        try:
            self._run_command(["tmux", "kill-session", "-t", self.tmux_session])
            self._output_buffer.append(f"[Killed tmux session {self.tmux_session}]")
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up tmux session and temp files without changing status."""
        try:
            # Only kill if session still exists (don't change status)
            self._run_command(["tmux", "kill-session", "-t", self.tmux_session])
        except Exception:
            pass
        # Clean up temp script file
        if self._script_path:
            try:
                os.unlink(self._script_path)
            except Exception:
                pass
            self._script_path = None
        # Clean up log file
        if self._log_path:
            try:
                os.unlink(self._log_path)
            except Exception:
                pass
            self._log_path = None


class SSHBackend(ExecutionBackend):
    """Backend for SSH remote execution with tmux for connection resilience."""

    def __init__(self, ssh_config: SSHConfig, task_id: int) -> None:
        if not PARAMIKO_AVAILABLE:
            raise ImportError("paramiko is required for SSH backend. Install with: pip install paramiko")

        self.ssh_config = ssh_config
        self.task_id = task_id
        self.client: Optional[Any] = None  # paramiko.SSHClient
        self.jump_client: Optional[Any] = None  # Jump host client for ProxyJump
        self.channel: Optional[Any] = None  # paramiko.Channel
        self._status = TaskStatus.PENDING
        self._output_buffer: list[str] = []
        self.tmux_session = f"orch_task_{task_id}"
        self._last_output_length = 0

    def _connect(self) -> None:
        """Establish SSH connection, with optional ProxyJump support."""
        if self.client is not None:
            try:
                # Test if connection is alive
                self.client.get_transport().send_ignore()
                return  # Connection is still good
            except Exception:
                # Connection is dead, close and reconnect
                try:
                    self.client.close()
                except Exception:
                    pass
                if self.jump_client:
                    try:
                        self.jump_client.close()
                    except Exception:
                        pass
                self.client = None
                self.jump_client = None

        # Handle ProxyJump if configured
        sock = None
        if self.ssh_config.proxy_jump:
            # Parse proxy_jump: "user@host" or "user@host:port"
            proxy_str = self.ssh_config.proxy_jump
            if "@" in proxy_str:
                proxy_user, proxy_host_part = proxy_str.split("@", 1)
            else:
                proxy_user = self.ssh_config.username
                proxy_host_part = proxy_str

            if ":" in proxy_host_part:
                proxy_host, proxy_port_str = proxy_host_part.split(":", 1)
                proxy_port = int(proxy_port_str)
            else:
                proxy_host = proxy_host_part
                proxy_port = 22

            # Connect to jump host first
            self.jump_client = paramiko.SSHClient()
            self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            jump_kwargs = {
                "hostname": proxy_host,
                "port": proxy_port,
                "username": proxy_user,
            }
            if self.ssh_config.key_file:
                jump_kwargs["key_filename"] = self.ssh_config.key_file
            elif self.ssh_config.password:
                jump_kwargs["password"] = self.ssh_config.password

            self.jump_client.connect(**jump_kwargs)

            # Create channel through jump host to final destination
            jump_transport = self.jump_client.get_transport()
            sock = jump_transport.open_channel(
                "direct-tcpip",
                (self.ssh_config.host, self.ssh_config.port),
                ("127.0.0.1", 0),
            )

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": self.ssh_config.host,
            "port": self.ssh_config.port,
            "username": self.ssh_config.username,
        }

        if sock:
            connect_kwargs["sock"] = sock

        if self.ssh_config.key_file:
            connect_kwargs["key_filename"] = self.ssh_config.key_file
        elif self.ssh_config.password:
            connect_kwargs["password"] = self.ssh_config.password

        self.client.connect(**connect_kwargs)

    def start(self, command: str, working_dir: str) -> None:
        """Execute command via SSH inside a tmux session for resilience."""
        self._status = TaskStatus.RUNNING

        try:
            self._connect()

            # Kill any existing tmux session with same name (cleanup from previous runs)
            self.client.exec_command(f"tmux kill-session -t {self.tmux_session} 2>/dev/null")
            time.sleep(0.1)

            # Create tmux session and run command inside it
            # The command runs in detached mode, we'll poll output via tmux capture-pane
            debug_command = f"echo '--- Start of Task (SSH) ---'; echo 'Intended Dir: {working_dir}'; echo 'Command: {command}'; echo '--- Output ---';"
            full_command = f"{debug_command} cd {working_dir} && {command}; echo '[TASK_EXIT_CODE:'$?']'"
            tmux_cmd = f'tmux new-session -d -s {self.tmux_session} "{full_command}"'

            stdin, stdout, stderr = self.client.exec_command(tmux_cmd)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error = stderr.read().decode("utf-8", errors="replace")
                self._status = TaskStatus.FAILED
                self._output_buffer.append(f"[tmux creation failed: {error}]")
            else:
                self._output_buffer.append(f"[Started tmux session: {self.tmux_session}]")

        except Exception as e:
            self._status = TaskStatus.FAILED
            self._output_buffer.append(f"SSH Error: {str(e)}")

    def poll_output(self) -> list[str]:
        """Get new output from tmux session (reconnection-safe)."""
        new_lines: list[str] = []

        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return new_lines

        try:
            self._connect()  # Reconnect if needed

            # Capture entire tmux pane output
            capture_cmd = f"tmux capture-pane -t {self.tmux_session} -p -S -"
            stdin, stdout, stderr = self.client.exec_command(capture_cmd)
            output = stdout.read().decode("utf-8", errors="replace")

            # Get only new lines since last poll
            all_lines = output.splitlines()
            if len(all_lines) > self._last_output_length:
                new_lines = all_lines[self._last_output_length:]
                self._output_buffer.extend(new_lines)
                self._last_output_length = len(all_lines)

                # Check for exit code marker
                for line in new_lines:
                    if "[TASK_EXIT_CODE:" in line:
                        try:
                            code = int(line.split(":")[1].rstrip("]"))
                            if code == 0:
                                self._status = TaskStatus.COMPLETED
                            else:
                                self._status = TaskStatus.FAILED
                            self._output_buffer.append(f"[Exit code: {code}]")
                        except (ValueError, IndexError):
                            pass

        except Exception as e:
            new_lines.append(f"[Poll Error - will retry: {str(e)}]")

        return new_lines

    def get_status(self) -> TaskStatus:
        """Check if tmux session is still running."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return self._status

        try:
            self._connect()

            # Check if tmux session still exists
            check_cmd = f"tmux has-session -t {self.tmux_session} 2>/dev/null && echo 'EXISTS' || echo 'GONE'"
            stdin, stdout, stderr = self.client.exec_command(check_cmd)
            result = stdout.read().decode("utf-8").strip()

            if result == "GONE":
                # Session ended, check if we captured exit code
                if self._status == TaskStatus.RUNNING:
                    self._status = TaskStatus.COMPLETED  # Assume success if session ended cleanly
                    self._output_buffer.append("[tmux session ended]")

        except Exception:
            pass  # Keep current status on connection error

        return self._status

    def kill(self) -> None:
        """Kill the tmux session."""
        self._status = TaskStatus.KILLED

        try:
            self._connect()
            self.client.exec_command(f"tmux kill-session -t {self.tmux_session}")
            self._output_buffer.append(f"[Killed tmux session {self.tmux_session}]")
        except Exception:
            pass

    def cleanup(self) -> None:
        """Close SSH connection (tmux session persists on remote)."""
        if self.client:
            try:
                # Clean up tmux session on remote
                self.client.exec_command(f"tmux kill-session -t {self.tmux_session} 2>/dev/null")
            except Exception:
                pass
            self.client.close()
            self.client = None
        if self.jump_client:
            try:
                self.jump_client.close()
            except Exception:
                pass
            self.jump_client = None


class SLURMBackend(ExecutionBackend):
    """Backend for SLURM cluster execution via salloc with tmux for resilience."""

    def __init__(self, slurm_config: SLURMConfig, task_id: int) -> None:
        if not PARAMIKO_AVAILABLE:
            raise ImportError("paramiko is required for SLURM backend. Install with: pip install paramiko")

        self.slurm_config = slurm_config
        self.task_id = task_id
        self.client: Optional[Any] = None
        self._status = TaskStatus.PENDING
        self._output_buffer: list[str] = []
        self.job_id: Optional[str] = None
        self.tmux_session = f"orch_slurm_{task_id}"
        self._last_output_length = 0

    def _connect(self) -> None:
        """Establish SSH connection to login node."""
        if self.client is not None:
            try:
                self.client.get_transport().send_ignore()
                return
            except Exception:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_cfg = self.slurm_config.ssh_config
        connect_kwargs = {
            "hostname": ssh_cfg.host,
            "port": ssh_cfg.port,
            "username": ssh_cfg.username,
        }

        if ssh_cfg.key_file:
            connect_kwargs["key_filename"] = ssh_cfg.key_file
        elif ssh_cfg.password:
            connect_kwargs["password"] = ssh_cfg.password

        self.client.connect(**connect_kwargs)

    def start(self, command: str, working_dir: str) -> None:
        """Execute command via salloc inside a tmux session for resilience."""
        self._status = TaskStatus.RUNNING

        try:
            self._connect()

            # Kill any existing tmux session
            self.client.exec_command(f"tmux kill-session -t {self.tmux_session} 2>/dev/null")
            time.sleep(0.1)

            # Build salloc command
            salloc_cmd = f"salloc --partition={self.slurm_config.partition} --time={self.slurm_config.time_limit}"
            if self.slurm_config.extra_flags:
                salloc_cmd += f" {self.slurm_config.extra_flags}"

            # Full command: salloc ... bash -c "cd dir && command"
            # We add exit code marker at the end
            debug_command = f"echo '--- Start of Task (SLURM) ---'; echo 'Intended Dir: {working_dir}'; echo 'Command: {command}'; echo '--- Output ---';"
            inner_command = f"{debug_command} cd '{working_dir}' && {command}"
            full_command = f'{salloc_cmd} bash -c "{inner_command}"; echo \'[TASK_EXIT_CODE:\'$?\']\''

            # Create tmux session with salloc command
            tmux_cmd = f"tmux new-session -d -s {self.tmux_session} '{full_command}'"

            stdin, stdout, stderr = self.client.exec_command(tmux_cmd)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error = stderr.read().decode("utf-8", errors="replace")
                self._status = TaskStatus.FAILED
                self._output_buffer.append(f"[tmux creation failed: {error}]")
            else:
                self._output_buffer.append(f"[SLURM] Started tmux session: {self.tmux_session}")
                self._output_buffer.append(f"[SLURM] Command: {salloc_cmd}")

        except Exception as e:
            self._status = TaskStatus.FAILED
            self._output_buffer.append(f"SLURM Error: {str(e)}")

    def poll_output(self) -> list[str]:
        """Get new output from SLURM job via tmux."""
        new_lines: list[str] = []

        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return new_lines

        try:
            self._connect()

            # Capture tmux pane output
            capture_cmd = f"tmux capture-pane -t {self.tmux_session} -p -S -"
            stdin, stdout, stderr = self.client.exec_command(capture_cmd)
            output = stdout.read().decode("utf-8", errors="replace")

            all_lines = output.splitlines()
            if len(all_lines) > self._last_output_length:
                new_lines = all_lines[self._last_output_length:]
                self._output_buffer.extend(new_lines)
                self._last_output_length = len(all_lines)

                # Parse for job ID and exit code
                for line in new_lines:
                    if "Granted job allocation" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            self.job_id = parts[3]
                            self._output_buffer.append(f"[SLURM Job ID: {self.job_id}]")

                    if "[TASK_EXIT_CODE:" in line:
                        try:
                            code = int(line.split(":")[1].rstrip("]"))
                            if code == 0:
                                self._status = TaskStatus.COMPLETED
                            else:
                                self._status = TaskStatus.FAILED
                            self._output_buffer.append(f"[Exit code: {code}]")
                        except (ValueError, IndexError):
                            pass

        except Exception as e:
            new_lines.append(f"[Poll Error - will retry: {str(e)}]")

        return new_lines

    def get_status(self) -> TaskStatus:
        """Check if SLURM job is still running."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return self._status

        try:
            self._connect()

            # Check if tmux session still exists
            check_cmd = f"tmux has-session -t {self.tmux_session} 2>/dev/null && echo 'EXISTS' || echo 'GONE'"
            stdin, stdout, stderr = self.client.exec_command(check_cmd)
            result = stdout.read().decode("utf-8").strip()

            if result == "GONE":
                if self._status == TaskStatus.RUNNING:
                    self._status = TaskStatus.COMPLETED
                    self._output_buffer.append("[tmux session ended]")

        except Exception:
            pass

        return self._status

    def kill(self) -> None:
        """Kill the SLURM job and tmux session."""
        self._status = TaskStatus.KILLED

        try:
            self._connect()

            # Cancel SLURM job if we have its ID
            if self.job_id:
                self.client.exec_command(f"scancel {self.job_id}")
                self._output_buffer.append(f"[Cancelled SLURM job {self.job_id}]")

            # Kill tmux session
            self.client.exec_command(f"tmux kill-session -t {self.tmux_session}")
            self._output_buffer.append(f"[Killed tmux session {self.tmux_session}]")

        except Exception:
            pass

    def cleanup(self) -> None:
        """Close SSH connection and clean up tmux session."""
        if self.client:
            try:
                self.client.exec_command(f"tmux kill-session -t {self.tmux_session} 2>/dev/null")
            except Exception:
                pass
            self.client.close()
            self.client = None


class Orchestrator:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Experiment Orchestrator")

        self.tasks: list[Task] = []
        self.task_counter: int = 0
        self.max_concurrent: int = 1
        self.update_queue: queue.Queue = queue.Queue()

        # Backend configuration
        self.current_backend_type = BackendType.LOCAL
        self.ssh_config = SSHConfig()
        self.slurm_config = SLURMConfig()
        self.current_remote_workdir: str = ""
        self.poll_interval: int = 5  # seconds
        self.state_file = "orchestrator_state.json"
        self.last_task_file_path: Optional[str] = None
        self.displayed_output_task_id: Optional[int] = None
        self.displayed_output_line_count: int = 0

        # Default Dropbox path (will be updated based on environment)
        self.dropbox_path: str = str(Path.home() / "Dropbox")

        # Position window on the monitor where cursor is (where command was run)
        self._position_on_cursor_monitor()

        self._setup_dark_theme()
        self._build_gui()
        self._load_state()  # Load previous state
        self._start_scheduler()
        self._start_update_loop()
        self._start_output_poller()

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _position_on_cursor_monitor(self) -> None:
        """Position the window on the monitor where the cursor currently is."""
        try:
            # Get cursor position using Win32 API
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                ]

            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_ulong),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", ctypes.c_ulong),
                ]

            user32 = ctypes.windll.user32

            # Get cursor position
            cursor = POINT()
            user32.GetCursorPos(ctypes.byref(cursor))

            # Get monitor from cursor position
            monitor_handle = user32.MonitorFromPoint(cursor, 2)  # MONITOR_DEFAULTTONEAREST

            # Get monitor info
            monitor_info = MONITORINFO()
            monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
            user32.GetMonitorInfoW(monitor_handle, ctypes.byref(monitor_info))

            # Position window on this monitor (centered-ish)
            win_x = monitor_info.rcWork.left + 50
            win_y = monitor_info.rcWork.top + 50
            self.root.geometry(f"1400x800+{win_x}+{win_y}")

        except Exception as e:
            print(f"Failed to position on cursor monitor: {e}")
            self.root.geometry("1400x800")

    def _setup_dark_theme(self) -> None:
        """Configure dark theme for the application."""
        # Dark color scheme
        bg_color = "#1e1e1e"  # Main background
        fg_color = "#d4d4d4"  # Main text
        accent_color = "#3c3c3c"  # Slightly lighter for frames
        entry_bg = "#2d2d2d"  # Entry background
        button_bg = "#404040"  # Button background
        select_bg = "#264f78"  # Selection background

        # Configure root window
        self.root.configure(bg=bg_color)

        # Create and configure style
        style = ttk.Style()
        style.theme_use("clam")  # clam theme is easier to customize

        # Configure general styles
        style.configure(".", background=bg_color, foreground=fg_color, fieldbackground=entry_bg)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color)
        style.configure("TButton", background=button_bg, foreground=fg_color)
        style.map("TButton", background=[("active", "#505050")])
        style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_color, insertcolor=fg_color)

        # Treeview styling
        style.configure(
            "Treeview",
            background=entry_bg,
            foreground=fg_color,
            fieldbackground=entry_bg,
            rowheight=25,
        )
        style.configure("Treeview.Heading", background=accent_color, foreground=fg_color)
        style.map("Treeview", background=[("selected", select_bg)], foreground=[("selected", "#ffffff")])

    def _build_gui(self) -> None:
        # Main frame with horizontal split
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left panel (main controls)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        main_frame.columnconfigure(0, weight=3)
        main_frame.rowconfigure(0, weight=1)

        # Right panel (output viewer)
        right_frame = ttk.LabelFrame(main_frame, text="Task Output", padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        main_frame.columnconfigure(1, weight=2)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        """Build the left panel with all controls."""
        parent.columnconfigure(0, weight=1)

        # Environment preset section
        env_frame = ttk.LabelFrame(parent, text="Environment", padding="5")
        env_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        preset_frame = ttk.Frame(env_frame)
        preset_frame.pack(fill=tk.X)
        ttk.Button(preset_frame, text="Local", command=self._preset_local).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="ginkgo", command=self._preset_ginkgo).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="rorqual", command=self._preset_rorqual).pack(side=tk.LEFT, padx=5)

        self.env_label = ttk.Label(env_frame, text="Current: Local")
        self.env_label.pack(pady=(5, 0))

        # SSH/SLURM configuration (hidden by default)
        self.config_frame = ttk.LabelFrame(parent, text="Remote Configuration", padding="5")
        self.config_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.config_frame.columnconfigure(1, weight=1)

        # SSH fields
        ttk.Label(self.config_frame, text="Password:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.ssh_pass_var = tk.StringVar()
        ttk.Entry(self.config_frame, textvariable=self.ssh_pass_var, show="*").grid(row=0, column=1, sticky="ew")

        # Remote working directory
        ttk.Label(self.config_frame, text="Remote Dir:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.remote_workdir_var = tk.StringVar()
        ttk.Entry(self.config_frame, textvariable=self.remote_workdir_var).grid(row=1, column=1, sticky="ew", pady=(5, 0))

        # SLURM specific fields
        self.slurm_fields_frame = ttk.Frame(self.config_frame)
        self.slurm_fields_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        ttk.Label(self.slurm_fields_frame, text="Partition:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.slurm_partition_var = tk.StringVar(value="default")
        ttk.Entry(self.slurm_fields_frame, textvariable=self.slurm_partition_var, width=15).grid(row=0, column=1, padx=(0, 10))

        ttk.Label(self.slurm_fields_frame, text="Time:").grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.slurm_time_var = tk.StringVar(value="1:00:00")
        ttk.Entry(self.slurm_fields_frame, textvariable=self.slurm_time_var, width=10).grid(row=0, column=3, padx=(0, 10))

        ttk.Label(self.slurm_fields_frame, text="Flags:").grid(row=0, column=4, sticky="w", padx=(0, 5))
        self.slurm_flags_var = tk.StringVar()
        ttk.Entry(self.slurm_fields_frame, textvariable=self.slurm_flags_var, width=20).grid(row=0, column=5)

        # Hide config frame by default (Local mode)
        self.config_frame.grid_remove()
        self.slurm_fields_frame.grid_remove()

        # Hidden SSH config fields (populated by presets)
        self.ssh_host_var = tk.StringVar()
        self.ssh_port_var = tk.StringVar(value="22")
        self.ssh_user_var = tk.StringVar()
        self.ssh_key_var = tk.StringVar()
        self.ssh_proxy_var = tk.StringVar()

        # Concurrency control section
        concurrency_frame = ttk.LabelFrame(parent, text="Concurrency", padding="5")
        concurrency_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(concurrency_frame, text="Max Concurrent:").grid(row=0, column=0, padx=5)
        self.concurrency_var = tk.StringVar(value="1")
        self.concurrency_entry = ttk.Entry(concurrency_frame, textvariable=self.concurrency_var, width=8)
        self.concurrency_entry.grid(row=0, column=1, padx=5)
        ttk.Button(concurrency_frame, text="Set", command=self._set_concurrency).grid(row=0, column=2, padx=5)

        self.concurrency_label = ttk.Label(concurrency_frame, text="Current: 1")
        self.concurrency_label.grid(row=0, column=3, padx=10)

        self.running_label = ttk.Label(concurrency_frame, text="Running: 0")
        self.running_label.grid(row=0, column=4, padx=10)

        # Load tasks section
        load_frame = ttk.LabelFrame(parent, text="Load Tasks", padding="5")
        load_frame.grid(row=3, column=0, sticky="ew", pady=(0, 5))

        ttk.Button(load_frame, text="Browse tasks.txt", command=self._load_tasks_from_file).pack(side=tk.LEFT, padx=5)

        # Task list section
        list_frame = ttk.LabelFrame(parent, text="Task Queue", padding="5")
        list_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 5))
        parent.rowconfigure(4, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Treeview for tasks
        columns = ("ID", "Backend", "Status", "Command")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("ID", text="ID")
        self.tree.heading("Backend", text="Backend")
        self.tree.heading("Status", text="Status")
        self.tree.heading("Command", text="Command")
        self.tree.column("ID", width=40, anchor="center")
        self.tree.column("Backend", width=80, anchor="center")
        self.tree.column("Status", width=80, anchor="center")
        self.tree.column("Command", width=400)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self._on_task_select)

        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=5, column=0, sticky="ew")

        ttk.Button(button_frame, text="Kill", command=self._kill_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Re-run Failed", command=self._rerun_failed).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Re-run Selected", command=self._rerun_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Remove Done", command=self._remove_completed).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear Pending", command=self._clear_pending).pack(side=tk.LEFT, padx=2)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=6, column=0, sticky="ew", pady=(5, 0))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        """Build the right panel with output viewer."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # Polling control
        poll_frame = ttk.Frame(parent)
        poll_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(poll_frame, text="Poll Interval (s):").pack(side=tk.LEFT)
        self.poll_interval_var = tk.StringVar(value="5")
        ttk.Entry(poll_frame, textvariable=self.poll_interval_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(poll_frame, text="Set", command=self._set_poll_interval).pack(side=tk.LEFT)
        ttk.Button(poll_frame, text="Clear Output", command=self._clear_output).pack(side=tk.LEFT, padx=(20, 0))

        # Output text area
        self.output_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.grid(row=1, column=0, sticky="nsew")

        # Configure text colors for dark theme
        self.output_text.configure(bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4")

        # Selected task label
        self.selected_task_var = tk.StringVar(value="No task selected")
        ttk.Label(parent, textvariable=self.selected_task_var).grid(row=2, column=0, sticky="w", pady=(5, 0))

    def _set_poll_interval(self) -> None:
        """Set the output polling interval."""
        try:
            interval = int(self.poll_interval_var.get())
            if interval < 1:
                raise ValueError("Must be at least 1")
            self.poll_interval = interval
            self.status_var.set(f"Poll interval set to {interval}s")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Enter a valid positive integer.\n{e}")

    def _clear_output(self) -> None:
        """Clear the output text area."""
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _on_task_select(self, event) -> None:
        """Handle task selection in treeview."""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            task_id = int(item["values"][0])
            self.selected_task_var.set(f"Selected: Task #{task_id}")

            # Show output for this task
            for task in self.tasks:
                if task.id == task_id:
                    print(f"[DEBUG] Selected task {task_id}, buffer has {len(task.output_buffer)} lines")
                    self._full_refresh_output_view(task)
                    self.displayed_output_task_id = task.id
                    self.displayed_output_line_count = len(task.output_buffer)
                    break
        else:
            self.selected_task_var.set("No task selected")
            self.displayed_output_task_id = None
            self.displayed_output_line_count = 0

    def _full_refresh_output_view(self, task: Task) -> None:
        """Display output buffer for a task."""
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)

        if task.output_buffer:
            lines_to_show = task.output_buffer[-500:]  # Show last 500 lines
            print(f"[DEBUG] Displaying {len(lines_to_show)} lines, first few: {lines_to_show[:3]}")
            # Insert all lines at once for efficiency
            text_content = "\n".join(lines_to_show) + "\n"
            self.output_text.insert(tk.END, text_content)
        else:
            self.output_text.insert(tk.END, "[No output yet]\n")

        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)
        # Force visual update
        self.output_text.update_idletasks()

    def _on_closing(self) -> None:
        """Handle window closing event."""
        self._save_state()
        self.root.destroy()

    def _save_state(self) -> None:
        """Save the current task list to a file."""
        state = {
            "tasks": [],
            "task_counter": self.task_counter,
            "last_task_file_path": self.last_task_file_path,
        }
        for task in self.tasks:
            task_data = {
                "id": task.id,
                "command": task.command,
                "status": task.status.value,
                "backend_type": task.backend_type.value,
                "remote_workdir": task.remote_workdir,
                "tmux_session": task.tmux_session,
                "output_buffer": task.output_buffer,
            }
            if task.ssh_config:
                task_data["ssh_config"] = task.ssh_config.__dict__
            if task.slurm_config:
                task_data["slurm_config"] = {
                    "partition": task.slurm_config.partition,
                    "time_limit": task.slurm_config.time_limit,
                    "extra_flags": task.slurm_config.extra_flags,
                    "ssh_config": task.slurm_config.ssh_config.__dict__,
                }
            state["tasks"].append(task_data)

        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
            self.status_var.set("State saved.")
        except Exception as e:
            self.status_var.set(f"Error saving state: {e}")

    def _load_state(self) -> None:
        """Load task list from a file."""
        if not os.path.exists(self.state_file):
            self.status_var.set("No previous state found.")
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.task_counter = state.get("task_counter", 0)
            self.last_task_file_path = state.get("last_task_file_path")
            loaded_tasks = []
            for task_data in state.get("tasks", []):
                status_val = task_data["status"]
                backend_val = task_data["backend_type"]

                task = Task(
                    id=task_data["id"],
                    command=task_data["command"],
                    status=TaskStatus(status_val),
                    backend_type=BackendType(backend_val),
                    remote_workdir=task_data.get("remote_workdir", ""),
                    tmux_session=task_data.get("tmux_session"),
                    output_buffer=task_data.get("output_buffer", []),
                )
                if "ssh_config" in task_data:
                    task.ssh_config = SSHConfig(**task_data["ssh_config"])
                if "slurm_config" in task_data:
                    slurm_data = task_data["slurm_config"]
                    slurm_data["ssh_config"] = SSHConfig(**slurm_data["ssh_config"])
                    task.slurm_config = SLURMConfig(**slurm_data)
                
                loaded_tasks.append(task)

            self.tasks = loaded_tasks
            self._reconnect_running_tasks()
            self._update_tree()
            self.status_var.set(f"Loaded {len(self.tasks)} tasks from previous session.")

        except Exception as e:
            self.status_var.set(f"Error loading state: {e}")
            messagebox.showerror("Load State Error", f"Failed to load state from {self.state_file}:\n{e}")

    def _reconnect_running_tasks(self) -> None:
        """Create backend instances and start monitoring for tasks that were running."""
        for task in self.tasks:
            if task.status == TaskStatus.RUNNING:
                try:
                    backend = None
                    if task.backend_type == BackendType.LOCAL:
                        backend = LocalBackend(task.id)
                        backend.tmux_session = task.tmux_session
                        task.backend = backend
                    elif task.backend_type == BackendType.SSH:
                        if not task.ssh_config:
                            raise ValueError("SSH config missing for running task")
                        backend = SSHBackend(task.ssh_config, task.id)
                        backend.tmux_session = task.tmux_session
                        task.backend = backend
                    elif task.backend_type == BackendType.SLURM:
                        if not task.slurm_config:
                            raise ValueError("SLURM config missing for running task")
                        backend = SLURMBackend(task.slurm_config, task.id)
                        backend.tmux_session = task.tmux_session
                        task.backend = backend
                    
                    # If a backend was created, start a thread to monitor it
                    if backend:
                        thread = threading.Thread(target=self._monitor_task, args=(task,), daemon=True)
                        thread.start()

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.output_buffer.append(f"[Reconnect failed: {e}]")


    def _get_current_ssh_config(self) -> SSHConfig:
        """Get SSH config from GUI fields."""
        return SSHConfig(
            host=self.ssh_host_var.get().strip(),
            port=int(self.ssh_port_var.get() or "22"),
            username=self.ssh_user_var.get().strip(),
            key_file=self.ssh_key_var.get().strip(),
            password=self.ssh_pass_var.get(),
            proxy_jump=self.ssh_proxy_var.get().strip(),
        )

    def _get_current_slurm_config(self) -> SLURMConfig:
        """Get SLURM config from GUI fields."""
        return SLURMConfig(
            ssh_config=self._get_current_ssh_config(),
            partition=self.slurm_partition_var.get().strip(),
            time_limit=self.slurm_time_var.get().strip(),
            extra_flags=self.slurm_flags_var.get().strip(),
        )

    def _preset_local(self) -> None:
        """Set preset for local execution."""
        self.current_backend_type = BackendType.LOCAL
        self.config_frame.grid_remove()
        self.slurm_fields_frame.grid_remove()
        self.env_label.config(text="Current: Local")
        self.status_var.set("Environment: Local")

    def _preset_ginkgo(self) -> None:
        """Set preset for ginkgo (lab machine via SSH)."""
        self.current_backend_type = BackendType.SSH
        self.ssh_host_var.set("ginkgo.criugm.qc.ca")
        self.ssh_port_var.set("22")
        self.ssh_user_var.set("mleclei")
        self.ssh_proxy_var.set("mleclei@elm.criugm.qc.ca")
        self.ssh_key_var.set("")
        self.config_frame.grid()
        self.slurm_fields_frame.grid_remove()
        self.env_label.config(text="Current: ginkgo (SSH)")
        self.status_var.set("Environment: ginkgo (SSH via elm)")
        if not PARAMIKO_AVAILABLE:
            messagebox.showwarning("Missing", "paramiko not installed. Run: pip install paramiko")

    def _preset_rorqual(self) -> None:
        """Set preset for rorqual (SLURM cluster)."""
        self.current_backend_type = BackendType.SLURM
        self.ssh_host_var.set("rorqual1.alliancecan.ca")
        self.ssh_port_var.set("22")
        self.ssh_user_var.set("mleclei")
        self.ssh_proxy_var.set("")
        self.ssh_key_var.set("")
        self.config_frame.grid()
        self.slurm_fields_frame.grid()
        self.env_label.config(text="Current: rorqual (SLURM)")
        self.status_var.set("Environment: rorqual (SLURM cluster)")
        if not PARAMIKO_AVAILABLE:
            messagebox.showwarning("Missing", "paramiko not installed. Run: pip install paramiko")

    def _set_concurrency(self) -> None:
        try:
            new_max = int(self.concurrency_var.get())
            if new_max < 1:
                raise ValueError("Must be at least 1")
            self.max_concurrent = new_max
            self.concurrency_label.config(text=f"Current: {new_max}")
            self.status_var.set(f"Max concurrency set to {new_max}")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter a valid positive integer.\n{e}")

    def _load_tasks_from_file(self) -> None:
        """Load tasks from a selected file, starting in the last used directory."""
        initial_dir = self.dropbox_path
        if self.last_task_file_path:
            initial_dir = str(Path(self.last_task_file_path).parent)

        file_path = filedialog.askopenfilename(
            title="Select tasks.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*")],
            initialdir=initial_dir,
        )
        if file_path:
            self.last_task_file_path = file_path
            self._load_tasks_file(file_path)

    def _load_tasks_file(self, file_path: str) -> None:
        """Load commands from a task list file.

        File format:
        - One command per line
        - Lines starting with # are comments
        - Empty lines are ignored
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Extract local working directory from file path
            local_workdir = str(Path(file_path).parent.resolve())

            # Validate remote config for SSH/SLURM
            if self.current_backend_type in (BackendType.SSH, BackendType.SLURM):
                if not self.remote_workdir_var.get().strip():
                    messagebox.showerror("Missing Remote Dir", "Please fill in the Remote Dir field first.")
                    return

            count = 0
            for line in lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                self.task_counter += 1
                task = Task(
                    id=self.task_counter,
                    command=line,
                    status=TaskStatus.PENDING,
                    backend_type=self.current_backend_type,
                )

                # Attach backend-specific config and working directory
                if self.current_backend_type == BackendType.LOCAL:
                    task.remote_workdir = local_workdir  # Use local path for local tasks
                elif self.current_backend_type == BackendType.SSH:
                    task.ssh_config = self._get_current_ssh_config()
                    task.remote_workdir = self.remote_workdir_var.get().strip()
                elif self.current_backend_type == BackendType.SLURM:
                    task.slurm_config = self._get_current_slurm_config()
                    task.remote_workdir = self.remote_workdir_var.get().strip()

                self.tasks.append(task)
                count += 1

            self._update_tree()
            env_name = {BackendType.LOCAL: "Local", BackendType.SSH: "ginkgo", BackendType.SLURM: "rorqual"}
            self.status_var.set(f"Loaded {count} tasks for {env_name.get(self.current_backend_type, 'Unknown')}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load tasks:\n{str(e)}")

    def _kill_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a task to kill.")
            return

        item = self.tree.item(selection[0])
        task_id = int(item["values"][0])

        for task in self.tasks:
            if task.id == task_id and task.status == TaskStatus.RUNNING:
                if task.backend:
                    task.backend.kill()
                    task.status = TaskStatus.KILLED
                    self.status_var.set(f"Killed task #{task_id}")
                    self._update_tree()
                return

        messagebox.showinfo("Cannot Kill", "Only running tasks can be killed.")

    def _remove_completed(self) -> None:
        self.tasks = [t for t in self.tasks if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED)]
        self._update_tree()
        self.status_var.set("Removed completed/failed/killed tasks")

    def _clear_pending(self) -> None:
        self.tasks = [t for t in self.tasks if t.status != TaskStatus.PENDING]
        self._update_tree()
        self.status_var.set("Cleared pending tasks")

    def _rerun_failed(self) -> None:
        """Re-queue all failed and killed tasks."""
        count = 0
        for task in self.tasks:
            if task.status in (TaskStatus.FAILED, TaskStatus.KILLED):
                task.status = TaskStatus.PENDING
                task.backend = None
                task.tmux_session = None
                count += 1
        self._update_tree()
        self.status_var.set(f"Re-queued {count} failed/killed tasks")

    def _rerun_selected(self) -> None:
        """Re-queue the selected task if it's failed, killed, or completed."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a task to re-run.")
            return

        item = self.tree.item(selection[0])
        task_id = int(item["values"][0])

        for task in self.tasks:
            if task.id == task_id:
                if task.status in (TaskStatus.FAILED, TaskStatus.KILLED, TaskStatus.COMPLETED):
                    task.status = TaskStatus.PENDING
                    task.backend = None
                    task.tmux_session = None
                    self._update_tree()
                    self.status_var.set(f"Re-queued task #{task_id}")
                else:
                    messagebox.showinfo("Cannot Re-run", "Only completed, failed, or killed tasks can be re-run.")
                return

        messagebox.showerror("Not Found", f"Task #{task_id} not found.")

    def _update_tree(self) -> None:
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Re-populate
        for task in self.tasks:
            tag = task.status.value.lower()
            backend_short = {"Local": "Local", "SSH Remote": "ginkgo", "SLURM Cluster": "rorqual"}.get(
                task.backend_type.value, task.backend_type.value
            )
            self.tree.insert(
                "", "end", values=(task.id, backend_short, task.status.value, task.command), tags=(tag,)
            )

        # Color coding (dark theme compatible)
        self.tree.tag_configure("pending", background="#2d2d2d", foreground="#888888")
        self.tree.tag_configure("running", background="#3d3d00", foreground="#ffd700")
        self.tree.tag_configure("completed", background="#1e3d1e", foreground="#90ee90")
        self.tree.tag_configure("failed", background="#3d1e1e", foreground="#ff6b6b")
        self.tree.tag_configure("killed", background="#3d2d1e", foreground="#ffa07a")

        # Update running count
        running_count = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)
        self.running_label.config(text=f"Running: {running_count}")

    def _monitor_task(self, task: Task):
        """Polls a running task until completion."""
        backend = task.backend
        if not backend:
            return

        try:
            # Poll for completion
            print(f"[DEBUG] Task {task.id}: Starting monitor loop")
            first_status = backend.get_status()
            print(f"[DEBUG] Task {task.id}: Initial status: {first_status}")

            while backend.get_status() == TaskStatus.RUNNING:
                new_lines = backend.poll_output()
                if new_lines:
                    task.output_buffer.extend(new_lines)
                    self.update_queue.put("output")  # Signal GUI to refresh if selected
                time.sleep(2.0)

            # Final poll and status update
            print(f"[DEBUG] Task {task.id}: Exited monitor loop, final status: {backend.get_status()}")
            new_lines = backend.poll_output()
            if new_lines:
                task.output_buffer.extend(new_lines)
            task.status = backend.get_status()

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.output_buffer.append(f"[Monitor Error: {str(e)}]")
        finally:
            # Clean up tmux session and temp files after task completes
            if backend:
                backend.cleanup()
            self.update_queue.put("update")  # Signal GUI to update tree

    def _start_task(self, task: Task) -> None:
        """Start a task using the appropriate backend."""
        task.status = TaskStatus.RUNNING
        self.update_queue.put("update")

        def run_task():
            try:
                # 1. Create and start the appropriate backend
                if task.backend_type == BackendType.LOCAL:
                    if not task.remote_workdir:
                        raise ValueError("Working directory not set for local task")
                    print(f"[DEBUG] Task {task.id}: Creating LocalBackend")
                    backend = LocalBackend(task.id)
                    task.backend = backend
                    task.tmux_session = backend.tmux_session
                    print(f"[DEBUG] Task {task.id}: Starting with command: {task.command}")
                    print(f"[DEBUG] Task {task.id}: Working dir: {task.remote_workdir}")
                    backend.start(task.command, task.remote_workdir)
                    # Transfer any startup messages to task buffer
                    print(f"[DEBUG] Task {task.id}: Backend status after start: {backend._status}")
                    print(f"[DEBUG] Task {task.id}: Backend output buffer: {backend._output_buffer}")
                    task.output_buffer.extend(backend._output_buffer)

                elif task.backend_type == BackendType.SSH:
                    if not task.ssh_config or not task.remote_workdir:
                        raise ValueError("SSH config or remote workdir not set")
                    backend = SSHBackend(task.ssh_config, task.id)
                    task.backend = backend
                    task.tmux_session = backend.tmux_session
                    backend.start(task.command, task.remote_workdir)
                    # Transfer any startup messages to task buffer
                    task.output_buffer.extend(backend._output_buffer)

                elif task.backend_type == BackendType.SLURM:
                    if not task.slurm_config or not task.remote_workdir:
                        raise ValueError("SLURM config or remote workdir not set")
                    backend = SLURMBackend(task.slurm_config, task.id)
                    task.backend = backend
                    task.tmux_session = backend.tmux_session
                    backend.start(task.command, task.remote_workdir)
                    # Transfer any startup messages to task buffer
                    task.output_buffer.extend(backend._output_buffer)
                else:
                    raise ValueError(f"Unknown backend type: {task.backend_type}")

                self.update_queue.put("output")  # Signal that we have startup output

                # 2. Monitor the task for completion
                self._monitor_task(task)

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.output_buffer.append(f"[Start Error: {str(e)}]")
                print(f"Task {task.id} start error: {e}")
                self.update_queue.put("update")

        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()


    def _start_scheduler(self) -> None:
        """Scheduler that checks for tasks to start."""
        def scheduler_loop():
            while True:
                running_count = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)

                if running_count < self.max_concurrent:
                    # Find next pending task
                    for task in self.tasks:
                        if task.status == TaskStatus.PENDING:
                            self._start_task(task)
                            break

                # Check every second
                threading.Event().wait(1.0)

        thread = threading.Thread(target=scheduler_loop, daemon=True)
        thread.start()

    def _start_update_loop(self) -> None:
        """Process updates from background threads."""
        def check_updates():
            try:
                while True:
                    msg = self.update_queue.get_nowait()
                    if msg == "update":
                        self._update_tree()
                    elif msg == "output":
                        # This message signals that new output is available
                        self._incremental_update_output_view()

            except queue.Empty:
                pass
            self.root.after(500, check_updates)

        self.root.after(500, check_updates)

    def _append_task_output(self, new_lines: list[str]):
        """Append new lines to the output widget."""
        self.output_text.configure(state=tk.NORMAL)
        for line in new_lines:
            self.output_text.insert(tk.END, line + "\n")
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _incremental_update_output_view(self):
        """Append new output lines for the currently selected task."""
        if self.displayed_output_task_id is None:
            return

        for task in self.tasks:
            if task.id == self.displayed_output_task_id:
                buffer_len = len(task.output_buffer)
                if buffer_len > self.displayed_output_line_count:
                    new_lines = task.output_buffer[self.displayed_output_line_count:]
                    self._append_task_output(new_lines)
                    self.displayed_output_line_count = buffer_len
                break

    def _start_output_poller(self) -> None:
        """Periodically check for new output for the selected task."""
        # The actual polling now happens in the _monitor_task threads.
        # This loop is no longer strictly necessary as the monitor thread
        # now puts an "output" message on the queue.
        # However, keeping a slower, periodic refresh can be a good fallback
        # in case a message is missed.
        def poll_loop():
            self._incremental_update_output_view()
            self.root.after(self.poll_interval * 1000, poll_loop)

        self.root.after(1000, poll_loop)


def main() -> None:
    root = tk.Tk()
    app = Orchestrator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
