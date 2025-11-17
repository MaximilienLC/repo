"""
GUI Orchestrator for managing parallel experiment runs.
Supports Local, SSH Remote, and SLURM Cluster execution backends.
"""

import ctypes
import json
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

# Try to import paramiko for SSH support
try:
    import paramiko

    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None  # type: ignore

from utils import *


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

        # Remote Dropbox paths for different environments
        self.remote_dropbox_paths = {
            "ginkgo": "/scratch/mleclei/Dropbox",
            "rorqual": "/scratch/mleclei/Dropbox",
        }
        self.current_remote_dropbox: str = ""  # Current remote Dropbox root

        # Flag to track if app is shutting down (to avoid cleanup on close)
        self._shutting_down = False

        # Position window on the monitor where cursor is (where command was run)
        self._position_on_cursor_monitor()

        self._setup_dark_theme()
        self._build_gui()
        self._load_state()  # Load previous state
        self._start_scheduler()
        self._start_update_loop()
        self._start_output_poller()

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _find_ssh_key(self) -> str:
        """Auto-detect SSH private key from ~/.ssh directory."""
        ssh_dir = Path.home() / ".ssh"
        # Common key file names in order of preference
        key_names = ["id_ed25519", "id_rsa", "id_ecdsa", "id_dsa"]
        for key_name in key_names:
            key_path = ssh_dir / key_name
            if key_path.exists():
                return str(key_path)
        return ""

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
            monitor_handle = user32.MonitorFromPoint(
                cursor, 2
            )  # MONITOR_DEFAULTTONEAREST

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
        style.configure(
            ".", background=bg_color, foreground=fg_color, fieldbackground=entry_bg
        )
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color)
        style.configure("TButton", background=button_bg, foreground=fg_color)
        style.map("TButton", background=[("active", "#505050")])
        style.configure(
            "TEntry",
            fieldbackground=entry_bg,
            foreground=fg_color,
            insertcolor=fg_color,
        )

        # Treeview styling
        style.configure(
            "Treeview",
            background=entry_bg,
            foreground=fg_color,
            fieldbackground=entry_bg,
            rowheight=25,
        )
        style.configure(
            "Treeview.Heading", background=accent_color, foreground=fg_color
        )
        style.map(
            "Treeview",
            background=[("selected", select_bg)],
            foreground=[("selected", "#ffffff")],
        )

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
        ttk.Button(preset_frame, text="local", command=self._preset_local).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(preset_frame, text="ginkgo", command=self._preset_ginkgo).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(preset_frame, text="rorqual", command=self._preset_rorqual).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(preset_frame, text="Connect", command=self._connect_to_remote).pack(
            side=tk.RIGHT, padx=5
        )

        self.env_label = ttk.Label(env_frame, text="Current: local")
        self.env_label.pack(pady=(5, 0))

        # Hidden SSH/SLURM config fields (populated by presets, no GUI needed)
        self.ssh_host_var = tk.StringVar()
        self.ssh_port_var = tk.StringVar(value="22")
        self.ssh_user_var = tk.StringVar()
        self.ssh_key_var = tk.StringVar()
        self.ssh_pass_var = tk.StringVar()
        self.ssh_proxy_var = tk.StringVar()
        self.remote_workdir_var = tk.StringVar()
        self.slurm_partition_var = tk.StringVar(value="default")
        self.slurm_time_var = tk.StringVar(value="1:00:00")
        self.slurm_flags_var = tk.StringVar()

        # Concurrency control section
        concurrency_frame = ttk.LabelFrame(parent, text="Concurrency", padding="5")
        concurrency_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(concurrency_frame, text="Max Concurrent:").grid(
            row=0, column=0, padx=5
        )
        self.concurrency_var = tk.StringVar(value="1")
        self.concurrency_entry = ttk.Entry(
            concurrency_frame, textvariable=self.concurrency_var, width=8
        )
        self.concurrency_entry.grid(row=0, column=1, padx=5)
        ttk.Button(concurrency_frame, text="Set", command=self._set_concurrency).grid(
            row=0, column=2, padx=5
        )

        self.concurrency_label = ttk.Label(concurrency_frame, text="Current: 1")
        self.concurrency_label.grid(row=0, column=3, padx=10)

        self.running_label = ttk.Label(concurrency_frame, text="Running: 0")
        self.running_label.grid(row=0, column=4, padx=10)

        # Load tasks section
        load_frame = ttk.LabelFrame(parent, text="Load Tasks", padding="5")
        load_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))

        ttk.Button(
            load_frame, text="Browse tasks.txt", command=self._load_tasks_from_file
        ).pack(side=tk.LEFT, padx=5)

        # Task list section
        list_frame = ttk.LabelFrame(parent, text="Task Queue", padding="5")
        list_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 5))
        parent.rowconfigure(3, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Treeview for tasks
        columns = ("ID", "Backend", "Status", "Command")
        self.tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", selectmode="extended"
        )
        self.tree.heading("ID", text="ID")
        self.tree.heading("Backend", text="Backend")
        self.tree.heading("Status", text="Status")
        self.tree.heading("Command", text="Command")
        self.tree.column("ID", width=40, anchor="center")
        self.tree.column("Backend", width=80, anchor="center")
        self.tree.column("Status", width=80, anchor="center")
        self.tree.column("Command", width=400)

        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self._on_task_select)

        # Keyboard shortcuts for multi-selection
        self.tree.bind("<Control-a>", self._select_all_tasks)
        self.tree.bind("<Control-A>", self._select_all_tasks)
        self.tree.bind("<Shift-Up>", self._extend_selection_up)
        self.tree.bind("<Shift-Down>", self._extend_selection_down)

        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, sticky="ew")

        ttk.Button(button_frame, text="Approve", command=self._approve_selected).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            button_frame, text="Clear Unapproved", command=self._clear_unapproved
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            button_frame, text="Remove Selected", command=self._remove_selected
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            button_frame, text="Remove All Done", command=self._remove_completed
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Re-run", command=self._rerun_tasks).pack(
            side=tk.LEFT, padx=2
        )

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            parent, textvariable=self.status_var, relief="sunken", anchor="w"
        )
        status_bar.grid(row=5, column=0, sticky="ew", pady=(5, 0))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        """Build the right panel with output viewer."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # Polling control
        poll_frame = ttk.Frame(parent)
        poll_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(poll_frame, text="Poll Interval (s):").pack(side=tk.LEFT)
        self.poll_interval_var = tk.StringVar(value="5")
        ttk.Entry(poll_frame, textvariable=self.poll_interval_var, width=5).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(poll_frame, text="Set", command=self._set_poll_interval).pack(
            side=tk.LEFT
        )
        ttk.Button(poll_frame, text="Clear Output", command=self._clear_output).pack(
            side=tk.LEFT, padx=(20, 0)
        )

        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, state=tk.DISABLED
        )
        self.output_text.grid(row=1, column=0, sticky="nsew")

        # Configure text colors for dark theme
        self.output_text.configure(
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4"
        )

        # Selected task label
        self.selected_task_var = tk.StringVar(value="No task selected")
        ttk.Label(parent, textvariable=self.selected_task_var).grid(
            row=2, column=0, sticky="w", pady=(5, 0)
        )

    def _set_poll_interval(self) -> None:
        """Set the output polling interval."""
        try:
            interval = int(self.poll_interval_var.get())
            if interval < 1:
                raise ValueError("Must be at least 1")
            self.poll_interval = interval
            self.status_var.set(f"Poll interval set to {interval}s")
        except ValueError as e:
            messagebox.showerror(
                "Invalid Input", f"Enter a valid positive integer.\n{e}"
            )

    def _clear_output(self) -> None:
        """Clear the output text area."""
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _on_task_select(self, event) -> None:
        """Handle task selection in treeview."""
        selection = self.tree.selection()
        if selection:
            if len(selection) == 1:
                item = self.tree.item(selection[0])
                task_id = int(item["values"][0])
                self.selected_task_var.set(f"Selected: Task #{task_id}")

                # Show output for this task
                for task in self.tasks:
                    if task.id == task_id:
                        print(
                            f"[DEBUG] Selected task {task_id}, buffer has {len(task.output_buffer)} lines"
                        )
                        self._full_refresh_output_view(task)
                        self.displayed_output_task_id = task.id
                        self.displayed_output_line_count = len(task.output_buffer)
                        break
            else:
                # Multiple selection
                task_ids = [int(self.tree.item(s)["values"][0]) for s in selection]
                self.selected_task_var.set(f"Selected: {len(selection)} tasks")
                # Clear output view for multiple selection
                self.displayed_output_task_id = None
                self.displayed_output_line_count = 0
                self._clear_output()
        else:
            self.selected_task_var.set("No task selected")
            self.displayed_output_task_id = None
            self.displayed_output_line_count = 0

    def _select_all_tasks(self, event=None) -> None:
        """Select all tasks in the treeview (Ctrl+A)."""
        all_items = self.tree.get_children()
        if all_items:
            self.tree.selection_set(all_items)
        return "break"  # Prevent default Ctrl+A behavior

    def _extend_selection_up(self, event=None) -> None:
        """Extend selection upward (Shift+Up)."""
        selection = self.tree.selection()
        all_items = self.tree.get_children()
        if not all_items:
            return "break"

        if not selection:
            # If no selection, select the last item
            self.tree.selection_set(all_items[-1])
            self.tree.focus(all_items[-1])
        else:
            # Find the topmost selected item and add the one above it
            first_selected_idx = min(all_items.index(s) for s in selection)
            if first_selected_idx > 0:
                new_item = all_items[first_selected_idx - 1]
                self.tree.selection_add(new_item)
                self.tree.focus(new_item)
                self.tree.see(new_item)
        return "break"

    def _extend_selection_down(self, event=None) -> None:
        """Extend selection downward (Shift+Down)."""
        selection = self.tree.selection()
        all_items = self.tree.get_children()
        if not all_items:
            return "break"

        if not selection:
            # If no selection, select the first item
            self.tree.selection_set(all_items[0])
            self.tree.focus(all_items[0])
        else:
            # Find the bottommost selected item and add the one below it
            last_selected_idx = max(all_items.index(s) for s in selection)
            if last_selected_idx < len(all_items) - 1:
                new_item = all_items[last_selected_idx + 1]
                self.tree.selection_add(new_item)
                self.tree.focus(new_item)
                self.tree.see(new_item)
        return "break"

    def _full_refresh_output_view(self, task: Task) -> None:
        """Display output buffer for a task."""
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)

        if task.output_buffer:
            lines_to_show = task.output_buffer[-500:]  # Show last 500 lines
            print(
                f"[DEBUG] Displaying {len(lines_to_show)} lines, first few: {lines_to_show[:3]}"
            )
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
        self._shutting_down = True  # Signal to monitor threads not to cleanup
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
            # Get log path from backend if available
            log_path = task.log_file_path
            if (
                task.backend
                and hasattr(task.backend, "_log_path")
                and task.backend._log_path
            ):
                log_path = task.backend._log_path

            task_data = {
                "id": task.id,
                "command": task.command,
                "status": task.status.value,
                "backend_type": task.backend_type.value,
                "remote_workdir": task.remote_workdir,
                "tmux_session": task.tmux_session,
                "log_file_path": log_path,
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
                    log_file_path=task_data.get("log_file_path"),
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
            self.status_var.set(
                f"Loaded {len(self.tasks)} tasks from previous session."
            )

        except Exception as e:
            self.status_var.set(f"Error loading state: {e}")
            messagebox.showerror(
                "Load State Error", f"Failed to load state from {self.state_file}:\n{e}"
            )

    def _reconnect_running_tasks(self) -> None:
        """Create backend instances and start monitoring for tasks that were running."""
        reconnected_count = 0
        failed_count = 0

        for task in self.tasks:
            if task.status == TaskStatus.RUNNING:
                try:
                    backend = None
                    session_exists = False

                    if task.backend_type == BackendType.LOCAL:
                        backend = LocalBackend(task.id)
                        # Use the saved tmux session name
                        if task.tmux_session:
                            backend.tmux_session = task.tmux_session
                        # Restore log file path if available
                        if task.log_file_path and os.path.exists(task.log_file_path):
                            backend._log_path = task.log_file_path
                        # Check if session still exists
                        ret_code, _, _ = backend._run_command(
                            ["tmux", "has-session", "-t", backend.tmux_session]
                        )
                        session_exists = ret_code == 0
                        if session_exists:
                            # Session exists, set backend to RUNNING state
                            backend._status = TaskStatus.RUNNING
                            task.backend = backend
                            task.output_buffer.append(
                                f"[Reconnected to tmux session: {backend.tmux_session}]"
                            )
                            if backend._log_path:
                                task.output_buffer.append(
                                    f"[Restored log file: {backend._log_path}]"
                                )

                    elif task.backend_type == BackendType.SSH:
                        if not task.ssh_config:
                            raise ValueError("SSH config missing for running task")
                        backend = SSHBackend(task.ssh_config, task.id)
                        if task.tmux_session:
                            backend.tmux_session = task.tmux_session
                        # For SSH, we'll trust that it's still running and let monitor check
                        backend._status = TaskStatus.RUNNING
                        task.backend = backend
                        session_exists = True
                        task.output_buffer.append(
                            f"[Reconnected to SSH session: {backend.tmux_session}]"
                        )

                    elif task.backend_type == BackendType.SLURM:
                        if not task.slurm_config:
                            raise ValueError("SLURM config missing for running task")
                        backend = SLURMBackend(task.slurm_config, task.id)
                        if task.tmux_session:
                            backend.tmux_session = task.tmux_session
                        # For SLURM, we'll trust that it's still running and let monitor check
                        backend._status = TaskStatus.RUNNING
                        task.backend = backend
                        session_exists = True
                        task.output_buffer.append(
                            f"[Reconnected to SLURM session: {backend.tmux_session}]"
                        )

                    # If session exists, start monitoring thread
                    if backend and session_exists:
                        thread = threading.Thread(
                            target=self._monitor_task, args=(task,), daemon=True
                        )
                        thread.start()
                        reconnected_count += 1
                    else:
                        # Session doesn't exist, mark as failed
                        task.status = TaskStatus.FAILED
                        task.output_buffer.append(
                            f"[Session {task.tmux_session} no longer exists]"
                        )
                        failed_count += 1

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.output_buffer.append(f"[Reconnect failed: {e}]")
                    failed_count += 1

        if reconnected_count > 0 or failed_count > 0:
            self.status_var.set(
                f"Reconnected {reconnected_count} tasks, {failed_count} sessions lost"
            )

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
        self.current_remote_dropbox = ""
        self.env_label.config(text="Current: local")
        self.status_var.set("Environment: local")

    def _preset_ginkgo(self) -> None:
        """Set preset for ginkgo (lab machine via SSH)."""
        self.current_backend_type = BackendType.SSH
        self.current_remote_dropbox = self.remote_dropbox_paths["ginkgo"]
        self.ssh_host_var.set("ginkgo.criugm.qc.ca")
        self.ssh_port_var.set("22")
        self.ssh_user_var.set("mleclei")
        self.ssh_proxy_var.set("mleclei@elm.criugm.qc.ca")
        self.ssh_key_var.set(self._find_ssh_key())
        self.remote_workdir_var.set("")  # Will be auto-filled when loading tasks
        self.env_label.config(text="Current: ginkgo (SSH)")
        key_status = "with key" if self.ssh_key_var.get() else "no key found"
        self.status_var.set(f"Environment: ginkgo (SSH via elm, {key_status})")
        if not PARAMIKO_AVAILABLE:
            messagebox.showwarning(
                "Missing", "paramiko not installed. Run: pip install paramiko"
            )

    def _preset_rorqual(self) -> None:
        """Set preset for rorqual (SLURM cluster)."""
        self.current_backend_type = BackendType.SLURM
        self.current_remote_dropbox = self.remote_dropbox_paths["rorqual"]
        self.ssh_host_var.set("rorqual1.alliancecan.ca")
        self.ssh_port_var.set("22")
        self.ssh_user_var.set("mleclei")
        self.ssh_proxy_var.set("")
        self.ssh_key_var.set(self._find_ssh_key())
        self.remote_workdir_var.set("")  # Will be auto-filled when loading tasks
        # Hardcoded SLURM settings for rorqual
        self.slurm_partition_var.set("")  # No partition specified, use default
        self.slurm_time_var.set("3:00:00")  # 3 hours default
        self.slurm_flags_var.set(
            "--gpus=h100:1 --account=rrg-pbellec --mem=124G --cpus-per-task=16"
        )
        self.env_label.config(text="Current: rorqual (SLURM)")
        key_status = "with key" if self.ssh_key_var.get() else "no key found"
        self.status_var.set(f"Environment: rorqual (SLURM cluster, {key_status})")
        if not PARAMIKO_AVAILABLE:
            messagebox.showwarning(
                "Missing", "paramiko not installed. Run: pip install paramiko"
            )

    def _connect_to_remote(self) -> None:
        """Establish persistent SSH connection with MFA support."""
        if self.current_backend_type == BackendType.LOCAL:
            messagebox.showinfo(
                "Local Mode", "No connection needed for local execution."
            )
            return

        host = self.ssh_host_var.get()
        user = self.ssh_user_var.get()
        port = self.ssh_port_var.get()
        proxy = self.ssh_proxy_var.get()
        key_file = self.ssh_key_var.get()

        if not host or not user:
            messagebox.showerror(
                "Missing Config",
                "Please select an environment first (ginkgo or rorqual).",
            )
            return

        # Create SSH config
        ssh_config = SSHConfig(
            host=host,
            port=int(port) if port else 22,
            username=user,
            key_file=key_file,
            proxy_jump=proxy,
        )

        # Check if already connected
        ssh_manager = SSHConnectionManager.get_instance()
        if ssh_manager.is_connected():
            messagebox.showinfo(
                "Already Connected", f"Already connected to {ssh_manager.host}"
            )
            return

        # Show instructions
        result = messagebox.askokcancel(
            "SSH Connection",
            f"Connecting to {host}...\n\n"
            "If MFA is required:\n"
            "1. Approve on your phone when prompted\n"
            "2. Wait for connection to establish\n\n"
            "Note: On Windows, this may NOT work with MFA.\n"
            "Consider using WSL (Windows Subsystem for Linux) which supports SSH multiplexing.\n\n"
            "Click OK to attempt connection.",
        )

        if not result:
            self.status_var.set("Connection cancelled")
            return

        self.status_var.set(f"Connecting to {host}... (this may take a moment)")
        self.root.update()

        # Run connection in background thread to not freeze GUI
        def connect_thread():
            success, message = ssh_manager.connect(ssh_config)

            # Update GUI from main thread
            def update_gui():
                if success:
                    self.status_var.set(f"Connected to {host}")
                    self.env_label.config(
                        text=f"Current: {self._get_env_name()} (Connected)"
                    )
                    messagebox.showinfo(
                        "Connected",
                        f"Successfully connected to {host}!\n\nAll tasks will use this connection.",
                    )
                else:
                    self.status_var.set(f"Connection failed")
                    messagebox.showerror(
                        "Connection Failed",
                        f"Failed to connect:\n{message}\n\n"
                        "This is likely because MFA is required and Windows SSH\n"
                        "doesn't support interactive prompts in background mode.\n\n"
                        "Workaround: Use Windows Subsystem for Linux (WSL)\n"
                        "or connect via VPN that bypasses MFA.",
                    )

            self.root.after(0, update_gui)

        thread = threading.Thread(target=connect_thread, daemon=True)
        thread.start()

    def _get_env_name(self) -> str:
        """Get current environment name."""
        if self.current_backend_type == BackendType.LOCAL:
            return "local"
        elif self.current_backend_type == BackendType.SSH:
            return "ginkgo"
        elif self.current_backend_type == BackendType.SLURM:
            return "rorqual"
        return "unknown"

    def _set_concurrency(self) -> None:
        try:
            new_max = int(self.concurrency_var.get())
            if new_max < 1:
                raise ValueError("Must be at least 1")
            self.max_concurrent = new_max
            self.concurrency_label.config(text=f"Current: {new_max}")
            self.status_var.set(f"Max concurrency set to {new_max}")
        except ValueError as e:
            messagebox.showerror(
                "Invalid Input", f"Please enter a valid positive integer.\n{e}"
            )

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

            # Auto-compute remote working directory for SSH/SLURM
            if self.current_backend_type in (BackendType.SSH, BackendType.SLURM):
                if (
                    self.current_remote_dropbox
                    and not self.remote_workdir_var.get().strip()
                ):
                    # Convert local Dropbox path to remote Dropbox path
                    # e.g., C:\Users\Max\Dropbox\repos\project -> /scratch/mleclei/Dropbox/repos/project
                    local_dropbox = Path(self.dropbox_path).resolve()
                    local_task_dir = Path(local_workdir).resolve()

                    try:
                        # Get the relative path from local Dropbox to task directory
                        relative_path = local_task_dir.relative_to(local_dropbox)
                        # Construct the remote path
                        remote_workdir = (
                            f"{self.current_remote_dropbox}/{relative_path.as_posix()}"
                        )
                        self.remote_workdir_var.set(remote_workdir)
                        self.status_var.set(f"Auto-set remote dir: {remote_workdir}")
                    except ValueError:
                        # Task file is not inside Dropbox folder
                        pass

                if not self.remote_workdir_var.get().strip():
                    messagebox.showerror(
                        "Missing Remote Dir",
                        "Please fill in the Remote Dir field first.",
                    )
                    return

            # Parse commands from file
            commands = []
            for line in lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                commands.append(line)

            if not commands:
                messagebox.showinfo("No Tasks", "No valid commands found in file.")
                return

            # Show task selection dialog
            self._show_task_selection_dialog(commands, local_workdir)

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load tasks:\n{str(e)}")

    def _show_task_selection_dialog(
        self, commands: list[str], local_workdir: str
    ) -> None:
        """Show a dialog to select which tasks to add."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Tasks to Add")
        dialog.geometry("800x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Apply dark theme
        dialog.configure(bg="#1e1e1e")

        # Main frame
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Info label
        info_text = f"Found {len(commands)} tasks. Select which ones to add:"
        ttk.Label(main_frame, text=info_text).pack(anchor="w", pady=(0, 10))

        # Create frame for listbox with checkboxes
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Canvas and scrollbar for checkbox list
        canvas = tk.Canvas(list_frame, bg="#2d2d2d", highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Create checkboxes for each command
        check_vars: list[tk.BooleanVar] = []
        for i, cmd in enumerate(commands):
            var = tk.BooleanVar(value=True)  # Default selected
            check_vars.append(var)

            # Truncate long commands for display
            display_cmd = cmd if len(cmd) <= 100 else cmd[:97] + "..."
            cb = ttk.Checkbutton(
                scrollable_frame, text=f"{i+1}. {display_cmd}", variable=var
            )
            cb.pack(anchor="w", pady=1)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        def select_all():
            for var in check_vars:
                var.set(True)

        def deselect_all():
            for var in check_vars:
                var.set(False)

        def add_selected():
            selected_commands = [
                cmd for cmd, var in zip(commands, check_vars) if var.get()
            ]
            if not selected_commands:
                messagebox.showwarning(
                    "No Selection", "Please select at least one task."
                )
                return

            # Add selected tasks
            count = 0
            for cmd in selected_commands:
                self.task_counter += 1
                task = Task(
                    id=self.task_counter,
                    command=cmd,
                    status=TaskStatus.UNAPPROVED,
                    backend_type=self.current_backend_type,
                )

                # Attach backend-specific config and working directory
                if self.current_backend_type == BackendType.LOCAL:
                    task.remote_workdir = local_workdir
                elif self.current_backend_type == BackendType.SSH:
                    task.ssh_config = self._get_current_ssh_config()
                    task.remote_workdir = self.remote_workdir_var.get().strip()
                elif self.current_backend_type == BackendType.SLURM:
                    task.slurm_config = self._get_current_slurm_config()
                    task.remote_workdir = self.remote_workdir_var.get().strip()

                self.tasks.append(task)
                count += 1

            self._update_tree()
            env_name = {
                BackendType.LOCAL: "local",
                BackendType.SSH: "ginkgo",
                BackendType.SLURM: "rorqual",
            }
            self.status_var.set(
                f"Added {count} tasks for {env_name.get(self.current_backend_type, 'Unknown')}"
            )

            # Clean up and close dialog
            canvas.unbind_all("<MouseWheel>")
            dialog.destroy()

        def cancel():
            canvas.unbind_all("<MouseWheel>")
            dialog.destroy()

        ttk.Button(button_frame, text="Select All", command=select_all).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(button_frame, text="Deselect All", command=deselect_all).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(button_frame, text="Add Selected", command=add_selected).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(
            side=tk.RIGHT, padx=2
        )

        # Center dialog on parent
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = (
            self.root.winfo_y()
            + (self.root.winfo_height() - dialog.winfo_height()) // 2
        )
        dialog.geometry(f"+{x}+{y}")

    def _approve_selected(self) -> None:
        """Approve the selected unapproved task(s) to make them pending."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select task(s) to approve.")
            return

        task_ids = [int(self.tree.item(s)["values"][0]) for s in selection]
        count = 0

        for task in self.tasks:
            if task.id in task_ids and task.status == TaskStatus.UNAPPROVED:
                task.status = TaskStatus.PENDING
                count += 1

        if count > 0:
            self._update_tree()
            self.status_var.set(f"Approved {count} task(s)")
        else:
            messagebox.showinfo("Cannot Approve", "No unapproved tasks in selection.")

    def _kill_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select task(s) to kill.")
            return

        task_ids = [int(self.tree.item(s)["values"][0]) for s in selection]
        count = 0

        for task in self.tasks:
            if task.id in task_ids and task.status == TaskStatus.RUNNING:
                if task.backend:
                    task.backend.kill()
                    task.status = TaskStatus.KILLED
                    count += 1

        if count > 0:
            self._update_tree()
            self.status_var.set(f"Killed {count} task(s)")
        else:
            messagebox.showinfo("Cannot Kill", "No running tasks in selection.")

    def _remove_completed(self) -> None:
        self.tasks = [
            t
            for t in self.tasks
            if t.status
            not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED)
        ]
        self._update_tree()
        self.status_var.set("Removed completed/failed/killed tasks")

    def _remove_selected(self) -> None:
        """Remove the selected task(s) regardless of their status."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select task(s) to remove.")
            return

        task_ids = set(int(self.tree.item(s)["values"][0]) for s in selection)
        count = 0

        # Kill running tasks first, then remove
        for task in self.tasks:
            if task.id in task_ids:
                if task.status == TaskStatus.RUNNING and task.backend:
                    task.backend.kill()

        # Remove all selected tasks
        original_count = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.id not in task_ids]
        count = original_count - len(self.tasks)

        self._update_tree()
        self.status_var.set(f"Removed {count} task(s)")

    def _clear_unapproved(self) -> None:
        self.tasks = [t for t in self.tasks if t.status != TaskStatus.UNAPPROVED]
        self._update_tree()
        self.status_var.set("Cleared unapproved tasks")

    def _rerun_tasks(self) -> None:
        """Re-queue selected task(s) if selected, otherwise all failed/killed tasks."""
        selection = self.tree.selection()
        if selection:
            # Re-run selected task(s)
            task_ids = [int(self.tree.item(s)["values"][0]) for s in selection]
            count = 0

            for task in self.tasks:
                if task.id in task_ids:
                    if task.status in (
                        TaskStatus.FAILED,
                        TaskStatus.KILLED,
                        TaskStatus.COMPLETED,
                    ):
                        task.status = TaskStatus.PENDING
                        task.backend = None
                        task.tmux_session = None
                        task.output_buffer = []
                        count += 1

            if count > 0:
                self._update_tree()
                self.status_var.set(f"Re-queued {count} task(s)")
            else:
                messagebox.showinfo(
                    "Cannot Re-run", "No completed/failed/killed tasks in selection."
                )
        else:
            # Re-run all failed/killed tasks
            count = 0
            for task in self.tasks:
                if task.status in (TaskStatus.FAILED, TaskStatus.KILLED):
                    task.status = TaskStatus.PENDING
                    task.backend = None
                    task.tmux_session = None
                    task.output_buffer = []
                    count += 1
            self._update_tree()
            self.status_var.set(f"Re-queued {count} failed/killed tasks")

    def _update_tree(self) -> None:
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Re-populate
        for task in self.tasks:
            tag = task.status.value.lower()
            backend_short = {
                "Local": "local",
                "SSH Remote": "ginkgo",
                "SLURM Cluster": "rorqual",
            }.get(task.backend_type.value, task.backend_type.value)
            self.tree.insert(
                "",
                "end",
                values=(task.id, backend_short, task.status.value, task.command),
                tags=(tag,),
            )

        # Color coding (dark theme compatible)
        self.tree.tag_configure(
            "unapproved", background="#2d2d3d", foreground="#9999cc"
        )
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

            while (
                backend.get_status() == TaskStatus.RUNNING and not self._shutting_down
            ):
                new_lines = backend.poll_output()
                if new_lines:
                    task.output_buffer.extend(new_lines)
                    self.update_queue.put("output")  # Signal GUI to refresh if selected
                time.sleep(2.0)

            # Final poll and status update (only if not shutting down)
            if not self._shutting_down:
                print(
                    f"[DEBUG] Task {task.id}: Exited monitor loop, final status: {backend.get_status()}"
                )
                new_lines = backend.poll_output()
                if new_lines:
                    task.output_buffer.extend(new_lines)
                task.status = backend.get_status()

        except Exception as e:
            if not self._shutting_down:
                task.status = TaskStatus.FAILED
                task.output_buffer.append(f"[Monitor Error: {str(e)}]")
        finally:
            # Only clean up if task actually completed (not if GUI is closing)
            if backend and not self._shutting_down:
                backend.cleanup()
            if not self._shutting_down:
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
                    print(
                        f"[DEBUG] Task {task.id}: Starting with command: {task.command}"
                    )
                    print(f"[DEBUG] Task {task.id}: Working dir: {task.remote_workdir}")
                    backend.start(task.command, task.remote_workdir)
                    # Save log file path to task for state persistence
                    task.log_file_path = backend._log_path
                    # Transfer any startup messages to task buffer
                    print(
                        f"[DEBUG] Task {task.id}: Backend status after start: {backend._status}"
                    )
                    print(
                        f"[DEBUG] Task {task.id}: Backend output buffer: {backend._output_buffer}"
                    )
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
                running_count = sum(
                    1 for t in self.tasks if t.status == TaskStatus.RUNNING
                )

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
                    new_lines = task.output_buffer[self.displayed_output_line_count :]
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
