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
    ssh_config: Optional[SSHConfig] = None
    slurm_config: Optional[SLURMConfig] = None
    remote_workdir: str = ""  # Working directory on remote machine (for SSH/SLURM)
    backend: Optional[Any] = None  # ExecutionBackend instance
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
    """Backend for SSH remote execution with tmux for connection resilience."""

    def __init__(self, ssh_config: SSHConfig, task_id: int) -> None:
        if not PARAMIKO_AVAILABLE:
            raise ImportError(
                "paramiko is required for SSH backend. Install with: pip install paramiko"
            )

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
            self.client.exec_command(
                f"tmux kill-session -t {self.tmux_session} 2>/dev/null"
            )
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
                self._output_buffer.append(
                    f"[Started tmux session: {self.tmux_session}]"
                )

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
                new_lines = all_lines[self._last_output_length :]
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
                    self._status = (
                        TaskStatus.COMPLETED
                    )  # Assume success if session ended cleanly
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
                self.client.exec_command(
                    f"tmux kill-session -t {self.tmux_session} 2>/dev/null"
                )
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


class SSHConnectionManager:
    """Manages a persistent SSH connection for sending commands."""

    _instance: Optional["SSHConnectionManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.connected = False
        self.host = ""
        self._cmd_lock = threading.Lock()
        self._output_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> "SSHConnectionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _output_reader(self):
        """Background thread to read SSH output."""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self._output_queue.put(line.rstrip())
                else:
                    break
        except Exception:
            pass

    def connect(self, ssh_config: SSHConfig) -> tuple[bool, str]:
        """Establish persistent SSH connection."""
        if self.connected and self.process and self.process.poll() is None:
            return True, "Already connected"

        # Disconnect any existing connection
        self.disconnect()

        # Build SSH command - use -T for no pseudo-terminal (cleaner output)
        cmd = ["ssh", "-T"]

        if ssh_config.key_file:
            cmd.extend(["-i", ssh_config.key_file])

        if ssh_config.proxy_jump:
            cmd.extend(["-J", ssh_config.proxy_jump])

        if ssh_config.port and ssh_config.port != 22:
            cmd.extend(["-p", str(ssh_config.port)])

        cmd.append(f"{ssh_config.username}@{ssh_config.host}")

        try:
            # Start SSH process with pipes
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            self.host = ssh_config.host

            # Start background reader thread
            self._reader_thread = threading.Thread(
                target=self._output_reader, daemon=True
            )
            self._reader_thread.start()

            # Wait a moment for connection
            time.sleep(2)

            # Check if process is still running
            if self.process.poll() is not None:
                return False, "SSH process terminated - MFA may be required"

            # Test connection
            test_result = self.run_command("echo CONNECTION_TEST_OK", timeout=10)
            if "CONNECTION_TEST_OK" in test_result[1]:
                self.connected = True
                return True, f"Connected to {ssh_config.host}"
            else:
                self.disconnect()
                return False, f"Connection test failed: {test_result[2]}"

        except Exception as e:
            self.disconnect()
            return False, f"Connection failed: {e}"

    def run_command(self, cmd: str, timeout: float = 30) -> tuple[int, str, str]:
        """Run a command through the persistent SSH connection."""
        if not self.process or self.process.poll() is not None:
            self.connected = False
            return -1, "", "Not connected"

        with self._cmd_lock:
            try:
                # Clear queue
                while not self._output_queue.empty():
                    try:
                        self._output_queue.get_nowait()
                    except queue.Empty:
                        break

                # Send command with unique marker
                marker = f"__END_{int(time.time() * 1000)}__"
                full_cmd = f"{cmd}; EXIT_CODE=$?; echo {marker} $EXIT_CODE\n"

                self.process.stdin.write(full_cmd)
                self.process.stdin.flush()

                # Read output until we see the marker
                output_lines = []
                start_time = time.time()

                while time.time() - start_time < timeout:
                    try:
                        line = self._output_queue.get(timeout=0.5)
                        if marker in line:
                            # Extract exit code
                            parts = line.strip().split()
                            exit_code = int(parts[-1]) if len(parts) > 1 else 0
                            return exit_code, "\n".join(output_lines), ""
                        output_lines.append(line)
                    except queue.Empty:
                        # Check if process died
                        if self.process.poll() is not None:
                            self.connected = False
                            return -1, "\n".join(output_lines), "Connection lost"
                        continue

                return -1, "\n".join(output_lines), "Command timeout"

            except Exception as e:
                return -1, "", str(e)

    def disconnect(self):
        """Close the SSH connection."""
        self.connected = False
        if self.process:
            try:
                self.process.stdin.write("exit\n")
                self.process.stdin.flush()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
        self.host = ""

    def is_connected(self) -> bool:
        """Check if connection is still alive."""
        if not self.connected or not self.process:
            return False
        if self.process.poll() is not None:
            self.connected = False
            return False
        return True


class SLURMBackend(ExecutionBackend):
    """Backend for SLURM cluster execution via salloc with tmux for resilience."""

    def __init__(self, slurm_config: SLURMConfig, task_id: int) -> None:
        self.slurm_config = slurm_config
        self.task_id = task_id
        self._status = TaskStatus.PENDING
        self._output_buffer: list[str] = []
        self.job_id: Optional[str] = None
        self.tmux_session = f"orch_slurm_{task_id}"
        self._last_output_length = 0
        self._ssh_manager = SSHConnectionManager.get_instance()

    def _run_remote_command(self, remote_cmd: str) -> tuple[int, str, str]:
        """Run a command on the remote host via persistent SSH connection."""
        if not self._ssh_manager.is_connected():
            return -1, "", "Not connected - click Connect button first"
        return self._ssh_manager.run_command(remote_cmd)

    def start(self, command: str, working_dir: str) -> None:
        """Execute command via salloc inside a tmux session for resilience."""
        self._status = TaskStatus.RUNNING

        try:
            # Kill any existing tmux session
            self._run_remote_command(
                f"tmux kill-session -t {self.tmux_session} 2>/dev/null"
            )
            time.sleep(0.1)

            # Build salloc command
            salloc_parts = ["salloc"]
            if self.slurm_config.partition:
                salloc_parts.append(f"--partition={self.slurm_config.partition}")
            salloc_parts.append(f"--time={self.slurm_config.time_limit}")
            if self.slurm_config.extra_flags:
                salloc_parts.append(self.slurm_config.extra_flags)
            salloc_cmd = " ".join(salloc_parts)

            # Full command: salloc ... bash -c "cd dir && command"
            # We add exit code marker at the end
            debug_command = f"echo '--- Start of Task (SLURM) ---'; echo 'Intended Dir: {working_dir}'; echo 'Command: {command}'; echo '--- Output ---';"
            inner_command = f"{debug_command} cd '{working_dir}' && {command}"
            full_command = f"{salloc_cmd} bash -c \"{inner_command}\"; echo '[TASK_EXIT_CODE:'$?']'"

            # Create tmux session with salloc command
            tmux_cmd = f"tmux new-session -d -s {self.tmux_session} '{full_command}'"

            ret_code, stdout, stderr = self._run_remote_command(tmux_cmd)

            if ret_code != 0:
                self._status = TaskStatus.FAILED
                self._output_buffer.append(f"[tmux creation failed: {stderr}]")
            else:
                self._output_buffer.append(
                    f"[SLURM] Started tmux session: {self.tmux_session}"
                )
                self._output_buffer.append(f"[SLURM] Command: {salloc_cmd}")

        except subprocess.TimeoutExpired:
            self._status = TaskStatus.FAILED
            self._output_buffer.append(
                "[SSH connection timeout - did you click Connect first?]"
            )
        except Exception as e:
            self._status = TaskStatus.FAILED
            self._output_buffer.append(f"SLURM Error: {str(e)}")

    def poll_output(self) -> list[str]:
        """Get new output from SLURM job via tmux."""
        new_lines: list[str] = []

        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
            return new_lines

        try:
            # Capture tmux pane output
            capture_cmd = f"tmux capture-pane -t {self.tmux_session} -p -S -"
            ret_code, stdout, stderr = self._run_remote_command(capture_cmd)

            if ret_code == 0:
                all_lines = stdout.splitlines()
                if len(all_lines) > self._last_output_length:
                    new_lines = all_lines[self._last_output_length :]
                    self._output_buffer.extend(new_lines)
                    self._last_output_length = len(all_lines)

                    # Parse for job ID and exit code
                    for line in new_lines:
                        if "Granted job allocation" in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                self.job_id = parts[3]
                                self._output_buffer.append(
                                    f"[SLURM Job ID: {self.job_id}]"
                                )

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
            # Check if tmux session still exists
            check_cmd = f"tmux has-session -t {self.tmux_session} 2>/dev/null && echo 'EXISTS' || echo 'GONE'"
            ret_code, stdout, stderr = self._run_remote_command(check_cmd)
            result = stdout.strip()

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
            # Cancel SLURM job if we have its ID
            if self.job_id:
                self._run_remote_command(f"scancel {self.job_id}")
                self._output_buffer.append(f"[Cancelled SLURM job {self.job_id}]")

            # Kill tmux session
            self._run_remote_command(f"tmux kill-session -t {self.tmux_session}")
            self._output_buffer.append(f"[Killed tmux session {self.tmux_session}]")

        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up tmux session on remote."""
        try:
            self._run_remote_command(
                f"tmux kill-session -t {self.tmux_session} 2>/dev/null"
            )
        except Exception:
            pass
