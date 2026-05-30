"""Robust subprocess execution utilities for experiment stages."""

import subprocess
import sys

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class FallbackConsole:
        def print(self, msg: str, *args, **kwargs) -> None:
            print(msg)
    console = FallbackConsole()  # type: ignore

def run_stage(command: list[str], stage_name: str, check: bool = True) -> int:
    """
    Run a command as an experiment stage with robust error handling.
    
    Args:
        command: The command list to execute.
        stage_name: Human-readable name of the stage.
        check: Whether to exit immediately on failure.
        
    Returns:
        The exit code of the process.
    """
    console.print(f"\n[bold blue]>>> Starting Stage: {stage_name}[/bold blue]")
    console.print(f"[dim]Command: {' '.join(command)}[/dim]\n")
    
    try:
        result = subprocess.run(command, check=check, text=True)
        console.print(f"\n[bold green]>>> Stage Completed: {stage_name}[/bold green]\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]>>> Stage Failed: {stage_name} (exit code {e.returncode})[/bold red]")
        if check:
            sys.exit(e.returncode)
        return e.returncode
    except KeyboardInterrupt:
        console.print(f"\n[bold red]>>> Stage Interrupted by User: {stage_name}[/bold red]")
        sys.exit(130)
