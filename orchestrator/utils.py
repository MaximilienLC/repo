"""
GUI Orchestrator utils.
"""

import os
import queue
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Try to import paramiko for SSH support
try:
    import paramiko

    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None  # type: ignore


class TaskStatus(Enum):
    UNAPPROVED = "Unapproved"
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
    activation_cmd: str = ""
    ssh_config: Optional[SSHConfig] = None
    slurm_config: Optional[SLURMConfig] = None
    remote_workdir: str = ""  # Working directory on remote machine (for SSH/SLURM)
    backend: Optional["ExecutionBackend"] = None  # ExecutionBackend instance
    output_buffer: list[str] = field(default_factory=list)
    last_poll_time: float = 0.0
    tmux_session: Optional[str] = None
    log_file_path: Optional[str] = None  # Log file path for local tasks


# ============================================================================
# Execution Backends
# ============================================================================


class ExecutionBackend(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    def start(self, task: "Task") -> None:
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

    def start(self, task: "Task") -> None:
        """Execute command locally inside a tmux session."""
        self._status = TaskStatus.RUNNING
        command = task.command
        working_dir = task.remote_workdir

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
            log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="orch_output_")
            os.close(log_fd)
            self._log_path = log_path
            win_log_path = log_path.replace("\\", "/")

            # Use Unix line endings (\n) for bash script
            # Redirect all output to log file so we can read it reliably
            script_lines = [
                "#!/bin/bash",
                f'exec > "{win_log_path}" 2>&1',  # Redirect stdout and stderr to log file
                'echo "--- Start of Task (Local) ---"',
                f'echo "Intended Dir: {win_workdir}"',
                f'echo "Command: {command}"',
                'echo "--- Output ---"',
                f'cd "{win_workdir}" && {command}',
                "exit_code=$?",
                'echo "[TASK_EXIT_CODE:$exit_code]"',
                "exit $exit_code",
            ]
            script_content = "\n".join(script_lines) + "\n"

            # Write script to temp file with Unix line endings
            script_fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="orch_task_")
            try:
                os.write(script_fd, script_content.encode("utf-8"))
                os.close(script_fd)

                # Use Windows paths with forward slashes (itmux bash understands these)
                win_script_path = script_path.replace("\\", "/")
                bash_win_path = self._bash_path.replace("\\", "/")

                # Create session that runs the script
                full_command = f'"{bash_win_path}" "{win_script_path}"'

                # Create session with remain-on-exit so we can capture output after command finishes
                tmux_cmd = [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    self.tmux_session,
                    "-x",
                    "200",
                    "-y",
                    "50",  # Set window size for better output capture
                    full_command,
                ]

                ret_code, stdout, stderr = self._run_command(tmux_cmd)
                print(
                    f"[DEBUG] tmux new-session ret_code={ret_code}, stdout={stdout.strip()}, stderr={stderr.strip()}"
                )

                if ret_code != 0:
                    # Double-check if session actually exists despite error
                    time.sleep(0.2)  # Give tmux time to create session
                    check_ret, check_out, check_err = self._run_command(
                        ["tmux", "has-session", "-t", self.tmux_session]
                    )
                    print(
                        f"[DEBUG] has-session check: ret={check_ret}, out={check_out.strip()}, err={check_err.strip()}"
                    )
                    if check_ret == 0:
                        # Session exists, ignore the error
                        print(
                            f"[DEBUG] Session {self.tmux_session} exists despite ret_code={ret_code}, proceeding"
                        )
                        self._output_buffer.append(
                            f"[Started tmux session: {self.tmux_session}]"
                        )
                        self._script_path = script_path
                    else:
                        self._status = TaskStatus.FAILED
                        self._output_buffer.append(
                            f"[tmux creation failed: ret={ret_code}, err={stderr}]"
                        )
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
                    self._output_buffer.append(
                        f"[Started tmux session: {self.tmux_session}]"
                    )
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
        """Get new output from the log file or tmux capture-pane."""
        new_lines: list[str] = []

        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return new_lines

        try:
            all_lines: list[str] = []

            if self._log_path:
                # Read from log file (normal case)
                try:
                    with open(
                        self._log_path, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        all_lines = f.read().splitlines()
                except FileNotFoundError:
                    pass  # Log file not created yet

            if not all_lines:
                # Fall back to tmux capture-pane (for reconnected sessions without log file)
                ret_code, stdout, stderr = self._run_command(
                    ["tmux", "capture-pane", "-t", self.tmux_session, "-p", "-S", "-"]
                )
                if ret_code == 0:
                    all_lines = stdout.splitlines()

            # Debug: show first poll output
            if self._last_output_length == 0 and all_lines:
                print(
                    f"[DEBUG] First output read for {self.tmux_session} ({len(all_lines)} lines total):"
                )
                # Show all non-empty lines
                non_empty = [
                    (i, line) for i, line in enumerate(all_lines) if line.strip()
                ]
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
                            self._status = (
                                TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
                            )
                            self._output_buffer.append(f"[Exit code: {code}]")
                        except (ValueError, IndexError):
                            pass

        except Exception as e:
            new_lines.append(f"[Poll Error: {str(e)}]")

        return new_lines

    def get_status(self) -> TaskStatus:
        """Check if the local tmux session is still running."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return self._status

        try:
            ret_code, stdout, stderr = self._run_command(
                ["tmux", "has-session", "-t", self.tmux_session]
            )
            print(
                f"[DEBUG] has-session for {self.tmux_session}: ret={ret_code}, stderr={stderr.strip()}"
            )
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
    """Backend for SSH remote execution using a local tmux session."""

    _tmux_path = r"C:\Users\Max\Documents\itmux\bin\tmux.exe"

    def __init__(self, ssh_config: SSHConfig, task_id: int) -> None:
        if not PARAMIKO_AVAILABLE:
            raise ImportError(
                "paramiko is required for SSH backend. Install with: pip install paramiko"
            )

        self.ssh_config = ssh_config
        self.task_id = task_id
        self._status = TaskStatus.PENDING
        self._output_buffer: list[str] = []
        self.local_tmux_session = f"orch_ssh_{task_id}"
        self.remote_tmux_session = f"orch_task_{task_id}"
        self.tmux_session = self.remote_tmux_session  # For compatibility

    def _run_command(self, command: list[str]) -> tuple[int, str, str]:
        """Run a local command and return exit code, stdout, and stderr."""
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

    def start(self, task: "Task") -> None:
        """Execute command via SSH by controlling a local tmux session."""
        self._status = TaskStatus.RUNNING
        command = task.command
        working_dir = task.remote_workdir
        activation_cmd = task.activation_cmd

        try:
            # Kill any existing local tmux session for this task
            self._run_command(["tmux", "kill-session", "-t", self.local_tmux_session])
            time.sleep(0.1)

            # Create local tmux session
            ret_code, _, stderr = self._run_command(
                ["tmux", "new-session", "-d", "-s", self.local_tmux_session]
            )
            if ret_code != 0:
                raise Exception(f"Failed to create local tmux session: {stderr}")
            self._output_buffer.append(
                f"[Started local tmux session: {self.local_tmux_session}]"
            )

            # Build SSH command
            ssh_cmd_parts = ["ssh"]
            if self.ssh_config.key_file:
                ssh_cmd_parts.extend(["-i", f"'{self.ssh_config.key_file}'"])
            if self.ssh_config.proxy_jump:
                ssh_cmd_parts.extend(["-J", self.ssh_config.proxy_jump])
            if self.ssh_config.port != 22:
                ssh_cmd_parts.extend(["-p", str(self.ssh_config.port)])
            ssh_cmd_parts.append(f"{self.ssh_config.username}@{self.ssh_config.host}")
            ssh_cmd_str = " ".join(ssh_cmd_parts)

            # Send SSH command to local tmux
            self._run_command(
                ["tmux", "send-keys", "-t", self.local_tmux_session, ssh_cmd_str, "C-m"]
            )
            self._output_buffer.append(
                f"[Sent SSH command to {self.local_tmux_session}]"
            )

            # Set tmux option on remote to keep pane alive after command finishes
            tmux_setup_cmd = "tmux start-server ; tmux set-option -g remain-on-exit on"
            self._run_command(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    self.local_tmux_session,
                    tmux_setup_cmd,
                    "C-m",
                ]
            )
            time.sleep(0.5)

            # Command to run on remote machine
            debug_command = f"echo '--- Start of Task (SSH) ---'; echo 'Intended Dir: {working_dir}'; echo 'Command: {command}'; echo '--- Output ---';"
            
            commands_to_run = []
            if activation_cmd:
                commands_to_run.append("echo '--- Activating Environment ---'")
                commands_to_run.append('echo "PATH before activation: $PATH"')
                commands_to_run.append(activation_cmd)
                commands_to_run.append("echo 'Activation command finished.'")
                commands_to_run.append('echo "PATH after activation: $PATH"')
                commands_to_run.append("which python")
            commands_to_run.append(command)

            inner_command = f"cd '{working_dir}' && {' && '.join(commands_to_run)}"
            full_command = (
                f'{debug_command} {inner_command}; echo "[TASK_EXIT_CODE:$?]"; exec bash'
            )

            # Escape single quotes for the shell
            escaped_full_command = full_command.replace("'", "'\\''")

            # Create remote tmux session with command
            # Use -A to create or attach
            remote_tmux_cmd = f"tmux new-session -A -s {self.remote_tmux_session} '{escaped_full_command}'"

            # Send remote tmux command
            self._run_command(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    self.local_tmux_session,
                    remote_tmux_cmd,
                    "C-m",
                ]
            )
            self._output_buffer.append(
                f"[Sent remote tmux command to create/attach to {self.remote_tmux_session}]"
            )

        except Exception as e:
            self._status = TaskStatus.FAILED
            self._output_buffer.append(f"SSH Start Error: {str(e)}")

    def poll_output(self) -> list[str]:
        """Get new output by capturing the local tmux pane."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return []

        try:
            # Capture entire local tmux pane output
            capture_cmd = [
                "tmux",
                "capture-pane",
                "-t",
                self.local_tmux_session,
                "-p",
                "-S",
                "-",
            ]
            ret_code, stdout, stderr = self._run_command(capture_cmd)

            if ret_code != 0:
                # Session might be gone, which is a final state.
                # get_status will handle the transition.
                return []

            all_lines = stdout.splitlines()

            # Check for exit code marker in the whole buffer, as the pane might have scrolled
            for line in reversed(all_lines):
                if "[TASK_EXIT_CODE:" in line:
                    try:
                        code = int(line.split(":")[-1].strip("[]"))
                        self._status = (
                            TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
                        )
                    except (ValueError, IndexError):
                        pass
                    break  # Found the latest exit code

            # Return the entire buffer. The orchestrator is responsible for diffing/redrawing.
            return all_lines

        except Exception as e:
            return [f"[Poll Error: {str(e)}]"]

    def get_status(self) -> TaskStatus:
        """Check if the local tmux session is still running."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return self._status

        try:
            # Check if local tmux session still exists
            ret_code, _, _ = self._run_command(
                ["tmux", "has-session", "-t", self.local_tmux_session]
            )

            if ret_code != 0:
                # Session ended. If we didn't capture an exit code, assume completion.
                if self._status == TaskStatus.RUNNING:
                    self._status = TaskStatus.COMPLETED
                    self._output_buffer.append("[Local tmux session ended]")

        except Exception:
            pass  # Keep current status on error

        return self._status

    def kill(self) -> None:
        """Kill the local tmux session, which terminates the SSH connection."""
        self._status = TaskStatus.KILLED
        try:
            self._run_command(["tmux", "kill-session", "-t", self.local_tmux_session])
            self._output_buffer.append(
                f"[Killed local tmux session {self.local_tmux_session}]"
            )
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up both remote and local tmux sessions."""
        try:
            # First, kill the remote tmux session by sending the command through the local session
            kill_remote_cmd = f"tmux kill-session -t {self.remote_tmux_session}"
            self._run_command(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    self.local_tmux_session,
                    kill_remote_cmd,
                    "C-m",
                ]
            )
            time.sleep(0.3)  # Give it a moment to execute
        except Exception:
            pass
        try:
            # Then kill the local tmux session
            self._run_command(["tmux", "kill-session", "-t", self.local_tmux_session])
        except Exception:
            pass
        self.client = None
        self.jump_client = None


class SLURMBackend(ExecutionBackend):
    """Backend for SLURM cluster execution via a local tmux session."""

    _tmux_path = r"C:\Users\Max\Documents\itmux\bin\tmux.exe"

    def __init__(self, slurm_config: SLURMConfig, task_id: int) -> None:
        self.slurm_config = slurm_config
        self.task_id = task_id
        self._status = TaskStatus.PENDING
        self._output_buffer: list[str] = []
        self.job_id: Optional[str] = None
        self.local_tmux_session = f"orch_slurm_{task_id}"
        self.remote_tmux_session = f"orch_task_{task_id}"
        self.tmux_session = self.remote_tmux_session  # For compatibility

    def _run_command(self, command: list[str]) -> tuple[int, str, str]:
        """Run a local command and return exit code, stdout, and stderr."""
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

    def start(self, task: "Task") -> None:
        """Execute command via salloc by controlling a local tmux session."""
        self._status = TaskStatus.RUNNING
        command = task.command
        working_dir = task.remote_workdir
        activation_cmd = task.activation_cmd

        try:
            # Kill any existing local tmux session
            self._run_command(["tmux", "kill-session", "-t", self.local_tmux_session])
            time.sleep(0.1)

            # Create local tmux session
            ret_code, _, stderr = self._run_command(
                ["tmux", "new-session", "-d", "-s", self.local_tmux_session]
            )
            if ret_code != 0:
                raise Exception(f"Failed to create local tmux session: {stderr}")
            self._output_buffer.append(
                f"[Started local tmux session: {self.local_tmux_session}]"
            )

            # Build SSH command
            ssh_config = self.slurm_config.ssh_config
            ssh_cmd_parts = ["ssh"]
            if ssh_config.key_file:
                ssh_cmd_parts.extend(["-i", f"'{ssh_config.key_file}'"])
            if ssh_config.proxy_jump:
                ssh_cmd_parts.extend(["-J", ssh_config.proxy_jump])
            ssh_cmd_parts.append(f"{ssh_config.username}@{ssh_config.host}")
            ssh_cmd_str = " ".join(ssh_cmd_parts)

            # Send SSH command
            self._run_command(
                ["tmux", "send-keys", "-t", self.local_tmux_session, ssh_cmd_str, "C-m"]
            )
            self._output_buffer.append(
                f"[Sent SSH command to {self.local_tmux_session}]"
            )

            # Wait for MFA prompt and send "1"
            time.sleep(5)
            self._run_command(
                ["tmux", "send-keys", "-t", self.local_tmux_session, "1", "C-m"]
            )
            self._output_buffer.append("[Sent '1' for MFA prompt]")
            time.sleep(5)  # Wait for connection to establish

            # Set tmux option on remote to keep pane alive after command finishes
            tmux_setup_cmd = "tmux start-server ; tmux set-option -g remain-on-exit on"
            self._run_command(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    self.local_tmux_session,
                    tmux_setup_cmd,
                    "C-m",
                ]
            )
            time.sleep(0.5)

            # Build salloc command
            salloc_parts = ["salloc"]
            if self.slurm_config.partition:
                salloc_parts.append(f"--partition={self.slurm_config.partition}")
            salloc_parts.append(f"--time={self.slurm_config.time_limit}")
            if self.slurm_config.extra_flags:
                salloc_parts.append(self.slurm_config.extra_flags)
            salloc_cmd = " ".join(salloc_parts)

            # Full command to run in remote tmux
            debug_command = f"echo '--- Start of Task (SLURM) ---'; echo 'Intended Dir: {working_dir}'; echo 'Command: {command}'; echo '--- Output ---';"

            commands_to_run = []
            if activation_cmd:
                commands_to_run.append("echo '--- Activating Environment ---'")
                commands_to_run.append('echo "PATH before activation: $PATH"')
                commands_to_run.append(activation_cmd)
                commands_to_run.append("echo 'Activation command finished.'")
                commands_to_run.append('echo "PATH after activation: $PATH"')
                commands_to_run.append("which python")
            commands_to_run.append(command)

            inner_command = f"cd '{working_dir}' && {' && '.join(commands_to_run)}"
            salloc_inner_cmd = f"{salloc_cmd} bash -c \\\"{inner_command}; echo \\\"[TASK_EXIT_CODE:$?]\\\"; exec bash\\\""
            full_command = f"{debug_command} {salloc_inner_cmd}"

            # Escape single quotes for the shell
            escaped_full_command = full_command.replace("'", "'\\''")

            # Create remote tmux session with salloc command
            remote_tmux_cmd = f"tmux new-session -A -s {self.remote_tmux_session} '{escaped_full_command}'"

            # Send remote tmux command
            self._run_command(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    self.local_tmux_session,
                    remote_tmux_cmd,
                    "C-m",
                ]
            )
            self._output_buffer.append(
                f"[Sent salloc command to remote tmux session {self.remote_tmux_session}]"
            )

        except Exception as e:
            self._status = TaskStatus.FAILED
            self._output_buffer.append(f"SLURM Start Error: {str(e)}")

    def poll_output(self) -> list[str]:
        """Get new output by capturing the local tmux pane."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return []

        try:
            # Capture entire local tmux pane output
            capture_cmd = [
                "tmux",
                "capture-pane",
                "-t",
                self.local_tmux_session,
                "-p",
                "-S",
                "-",
            ]
            ret_code, stdout, stderr = self._run_command(capture_cmd)

            if ret_code != 0:
                return []

            all_lines = stdout.splitlines()

            # Parse for job ID and exit code from the whole buffer
            for line in reversed(all_lines):
                if self.job_id is None and "Granted job allocation" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        self.job_id = parts[3]

                if "[TASK_EXIT_CODE:" in line:
                    try:
                        code = int(line.split(":")[-1].strip("[]"))
                        self._status = (
                            TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
                        )
                    except (ValueError, IndexError):
                        pass
                    break  # Found the latest exit code

            return all_lines

        except Exception as e:
            return [f"[Poll Error: {str(e)}]"]

    def get_status(self) -> TaskStatus:
        """Check if the local tmux session is still running."""
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return self._status

        try:
            ret_code, _, _ = self._run_command(
                ["tmux", "has-session", "-t", self.local_tmux_session]
            )
            if ret_code != 0:
                if self._status == TaskStatus.RUNNING:
                    self._status = TaskStatus.COMPLETED
                    self._output_buffer.append("[Local tmux session ended]")
        except Exception:
            pass

        return self._status

    def kill(self) -> None:
        """Kill the local tmux session."""
        self._status = TaskStatus.KILLED
        try:
            # The remote salloc job will be killed when the SSH session terminates
            self._run_command(["tmux", "kill-session", "-t", self.local_tmux_session])
            self._output_buffer.append(
                f"[Killed local tmux session {self.local_tmux_session}]"
            )
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up both remote and local tmux sessions."""
        try:
            # First, kill the remote tmux session by sending the command through the local session
            kill_remote_cmd = f"tmux kill-session -t {self.remote_tmux_session}"
            self._run_command(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    self.local_tmux_session,
                    kill_remote_cmd,
                    "C-m",
                ]
            )
            time.sleep(0.3)  # Give it a moment to execute
        except Exception:
            pass
        try:
            # Then kill the local tmux session
            self._run_command(["tmux", "kill-session", "-t", self.local_tmux_session])
        except Exception:
            pass

