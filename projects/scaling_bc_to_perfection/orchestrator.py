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
    process: Optional[subprocess.Popen] = None
    cascade_slot: Optional[int] = None  # Position in cascade (local only)
    ssh_config: Optional[SSHConfig] = None
    slurm_config: Optional[SLURMConfig] = None
    remote_workdir: str = ""  # Working directory on remote machine (for SSH/SLURM)
    backend: Optional[Any] = None  # ExecutionBackend instance
    output_buffer: list[str] = field(default_factory=list)
    last_poll_time: float = 0.0


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
    """Backend for local execution with captured output."""

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        self.process: Optional[subprocess.Popen] = None
        self._status = TaskStatus.PENDING
        self._output_buffer: list[str] = []
        self._reader_thread: Optional[threading.Thread] = None

    def start(self, command: str, working_dir: str) -> None:
        """Start task with captured stdout/stderr."""
        self._status = TaskStatus.RUNNING

        self.process = subprocess.Popen(
            ["powershell", "-Command", command],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Start thread to read output
        def read_output():
            if self.process and self.process.stdout:
                for line in self.process.stdout:
                    self._output_buffer.append(line.rstrip())

        self._reader_thread = threading.Thread(target=read_output, daemon=True)
        self._reader_thread.start()

    def poll_output(self) -> list[str]:
        """Get all buffered output."""
        return self._output_buffer.copy()

    def get_status(self) -> TaskStatus:
        """Check if process is still running."""
        if self._status == TaskStatus.KILLED:
            return TaskStatus.KILLED

        if self.process is None:
            return self._status

        return_code = self.process.poll()
        if return_code is None:
            return TaskStatus.RUNNING
        elif return_code == 0:
            self._status = TaskStatus.COMPLETED
        else:
            self._status = TaskStatus.FAILED
        return self._status

    def kill(self) -> None:
        """Terminate the process."""
        if self.process:
            self.process.terminate()
            self._status = TaskStatus.KILLED

    def cleanup(self) -> None:
        """No cleanup needed for local backend."""
        pass


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
            full_command = f"cd {working_dir} && {command}; echo '[TASK_EXIT_CODE:'$?']'"
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
            full_command = f'{salloc_cmd} bash -c "cd {working_dir} && {command}"; echo \'[TASK_EXIT_CODE:\'$?\']\''

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

        # Default Dropbox path (will be updated based on environment)
        self.dropbox_path: str = str(Path.home() / "Dropbox")

        # Position window on the monitor where cursor is (where command was run)
        self._position_on_cursor_monitor()

        self._setup_dark_theme()
        self._build_gui()
        self._start_scheduler()
        self._start_update_loop()
        self._start_output_poller()

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
                    self._display_task_output(task)
                    break
        else:
            self.selected_task_var.set("No task selected")

    def _display_task_output(self, task: Task) -> None:
        """Display output buffer for a task."""
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)

        if task.output_buffer:
            for line in task.output_buffer[-500:]:  # Show last 500 lines
                self.output_text.insert(tk.END, line + "\n")
        else:
            self.output_text.insert(tk.END, "[No output yet]\n")

        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

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
        """Load tasks from a selected file (starts in Dropbox folder)."""
        file_path = filedialog.askopenfilename(
            title="Select tasks.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*")],
            initialdir=self.dropbox_path,
        )
        if file_path:
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
                elif task.process:  # Fallback for old local tasks
                    task.process.terminate()
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
                task.process = None
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
                    task.process = None
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

    def _start_task(self, task: Task) -> None:
        """Start a task using the appropriate backend."""
        task.status = TaskStatus.RUNNING
        self.update_queue.put("update")

        def run_task():
            try:
                # Create the appropriate backend
                if task.backend_type == BackendType.LOCAL:
                    if not task.remote_workdir:
                        raise ValueError("Working directory not set for local task")

                    backend = LocalBackend(task.id)
                    task.backend = backend
                    backend.start(task.command, task.remote_workdir)

                    # Poll for completion and update output buffer
                    while backend.get_status() == TaskStatus.RUNNING:
                        task.output_buffer = backend.poll_output()
                        time.sleep(1.0)

                    task.output_buffer = backend.poll_output()  # Final poll
                    task.status = backend.get_status()
                    backend.cleanup()

                elif task.backend_type == BackendType.SSH:
                    if not task.ssh_config:
                        raise ValueError("SSH config not set for SSH task")
                    if not task.remote_workdir:
                        raise ValueError("Remote working directory not set for SSH task")

                    backend = SSHBackend(task.ssh_config, task.id)
                    task.backend = backend
                    backend.start(task.command, task.remote_workdir)

                    # Poll for completion
                    while backend.get_status() == TaskStatus.RUNNING:
                        new_lines = backend.poll_output()
                        task.output_buffer.extend(new_lines)
                        time.sleep(2.0)

                    task.status = backend.get_status()
                    backend.cleanup()

                elif task.backend_type == BackendType.SLURM:
                    if not task.slurm_config:
                        raise ValueError("SLURM config not set for SLURM task")
                    if not task.remote_workdir:
                        raise ValueError("Remote working directory not set for SLURM task")

                    backend = SLURMBackend(task.slurm_config, task.id)
                    task.backend = backend
                    backend.start(task.command, task.remote_workdir)

                    # Poll for completion
                    while backend.get_status() == TaskStatus.RUNNING:
                        new_lines = backend.poll_output()
                        task.output_buffer.extend(new_lines)
                        time.sleep(2.0)

                    task.status = backend.get_status()
                    backend.cleanup()

                else:
                    raise ValueError(f"Unknown backend type: {task.backend_type}")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.output_buffer.append(f"[Error: {str(e)}]")
                print(f"Task {task.id} error: {e}")

            # Signal update
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
                    self.update_queue.get_nowait()
                    self._update_tree()
            except queue.Empty:
                pass
            self.root.after(500, check_updates)

        self.root.after(500, check_updates)

    def _start_output_poller(self) -> None:
        """Periodically update the output viewer for the selected task."""
        def poll_output():
            # Check if a task is selected
            selection = self.tree.selection()
            if selection:
                item = self.tree.item(selection[0])
                task_id = int(item["values"][0])

                # Find the task and refresh output display
                for task in self.tasks:
                    if task.id == task_id:
                        # Refresh if task is running (output is being updated)
                        if task.status == TaskStatus.RUNNING:
                            self._display_task_output(task)
                        break

            # Schedule next poll based on interval
            self.root.after(self.poll_interval * 1000, poll_output)

        self.root.after(self.poll_interval * 1000, poll_output)


def main() -> None:
    root = tk.Tk()
    app = Orchestrator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
