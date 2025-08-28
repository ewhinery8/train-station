#!/usr/bin/env python3
"""
Performance Scatter Plot Generator

This script loads JSON performance results from any operation and generates
a scatter plot comparing Train Station vs LibTorch performance across different
tensor shapes and sizes.

Usage:
    python performance_scatter.py <json_file> [output_file]

Example:
    python performance_scatter.py add_performance.json add_performance_plot.html
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_performance_data(json_file: str) -> List[Dict]:
    """
    Load performance data from JSON file.

    Args:
        json_file: Path to the JSON file containing performance results

    Returns:
        List of performance result dictionaries

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Performance file not found: {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON array of performance results")

    return data


def extract_operation_name(data: List[Dict]) -> str:
    """
    Extract operation name from performance data.

    Args:
        data: List of performance result dictionaries

    Returns:
        Operation name (e.g., "add_tensor", "mul_scalar")
    """
    if not data:
        return "unknown_operation"

    # Get the first operation name and clean it up
    operation = data[0].get("operation", "unknown")

    # Convert to title case for display
    if "_" in operation:
        # Handle cases like "add_tensor" -> "Add Tensor"
        parts = operation.split("_")
        return " ".join(part.capitalize() for part in parts)
    else:
        return operation.capitalize()


def extract_hardware_info(data: List[Dict]) -> str:
    """
    Extract hardware information from performance data.

    Args:
        data: List of performance result dictionaries

    Returns:
        Hardware information string or empty string if not available
    """
    if not data:
        return ""

    # Look for hardware info in the first result
    first_result = data[0]

    # Check for common hardware fields
    hardware_fields = ["hardware", "cpu", "processor", "system", "platform"]

    for field in hardware_fields:
        if field in first_result:
            return str(first_result[field])

    # If no hardware info found, try to get detailed system info
    try:
        import os
        import platform
        import subprocess

        # Basic system info
        system_info = f"{platform.system()} {platform.release()}"
        if platform.machine():
            system_info += f" ({platform.machine()})"

        # Try to get CPU information
        cpu_info = ""
        try:
            if platform.system() == "Linux":
                # Read CPU info from /proc/cpuinfo
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    for line in cpuinfo.split("\n"):
                        if line.startswith("model name"):
                            cpu_model = line.split(":")[1].strip()
                            cpu_info = f" | CPU: {cpu_model}"
                            break
                        elif line.startswith("Hardware"):
                            cpu_model = line.split(":")[1].strip()
                            cpu_info = f" | CPU: {cpu_model}"
                            break
            elif platform.system() == "Darwin":  # macOS
                # Use sysctl to get CPU info
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    cpu_info = f" | CPU: {result.stdout.strip()}"
            elif platform.system() == "Windows":
                # Use wmic to get CPU info
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        cpu_name = lines[1].strip()
                        if cpu_name:
                            cpu_info = f" | CPU: {cpu_name}"
        except Exception:
            pass

        # Try to get memory information
        memory_info = ""
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    for line in meminfo.split("\n"):
                        if line.startswith("MemTotal"):
                            mem_kb = int(line.split()[1])
                            mem_gb = mem_kb / (1024 * 1024)
                            memory_info = f" | RAM: {mem_gb:.1f}GB"
                            break
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.strip())
                    mem_gb = mem_bytes / (1024**3)
                    memory_info = f" | RAM: {mem_gb:.1f}GB"
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        mem_bytes = int(lines[1].strip())
                        mem_gb = mem_bytes / (1024**3)
                        memory_info = f" | RAM: {mem_gb:.1f}GB"
        except Exception:
            pass

        # Combine all information
        full_info = system_info + cpu_info + memory_info
        return full_info

    except ImportError:
        pass

    # If no hardware info found, return empty string
    return ""


def calculate_tensor_size(shape: List[int]) -> int:
    """
    Calculate total number of elements in a tensor.

    Args:
        shape: List of tensor dimensions

    Returns:
        Total number of elements
    """
    return np.prod(shape)


def prepare_plot_data(
    data: List[Dict],
) -> Tuple[List[float], List[float], List[str], List[str]]:
    """
    Prepare data for plotting.

    Args:
        data: List of performance result dictionaries

    Returns:
        Tuple of (train_station_times, libtorch_times, shapes, operations)
    """
    train_station_times = []
    libtorch_times = []
    shapes = []
    operations = []

    for result in data:
        # Convert nanoseconds to milliseconds for better readability
        train_time_ms = result.get("our_avg_time_ns", 0) / 1_000_000.0
        libtorch_time_ms = result.get("libtorch_avg_time_ns", 0) / 1_000_000.0

        train_station_times.append(train_time_ms)
        libtorch_times.append(libtorch_time_ms)

        # Format shape for display
        shape_str = str(result.get("shape", []))
        shapes.append(shape_str)

        # Get operation name
        operation = result.get("operation", "unknown")
        operations.append(operation)

    return train_station_times, libtorch_times, shapes, operations


def create_performance_scatter_plot(
    data: List[Dict], output_file: Optional[str] = None
) -> go.Figure:
    """
    Create a scatter plot comparing Train Station vs LibTorch performance.

    Args:
        data: List of performance result dictionaries
        output_file: Optional output file path

    Returns:
        Plotly figure object
    """
    if not data:
        raise ValueError("No performance data provided")

    # Extract operation name for title
    operation_name = extract_operation_name(data)

    # Calculate test statistics for title
    total_iterations = sum(result.get("iterations", 0) for result in data)
    avg_iterations = total_iterations // len(data) if data else 0
    total_warmup = (
        sum(result.get("warmup_iterations", 0) for result in data)
        if "warmup_iterations" in data[0]
        else 0
    )
    avg_warmup = total_warmup // len(data) if data and total_warmup > 0 else 0

    # Prepare data for plotting
    train_times, libtorch_times, shapes, operations = prepare_plot_data(data)

    # Calculate tensor sizes for x-axis
    tensor_sizes = [calculate_tensor_size(result.get("shape", [])) for result in data]

    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{operation_name} Performance Comparison",
            "Speedup (Train Station / LibTorch (C++ PyTorch backend))",
        ),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Add scatter plot for performance comparison (pure scatter, no lines)
    fig.add_trace(
        go.Scatter(
            x=tensor_sizes,
            y=train_times,
            mode="markers",
            name="Train Station",
            marker=dict(size=8, color="blue", symbol="circle"),
            hovertemplate=(
                "<b>Train Station</b><br>"
                + "Tensor Size: %{x}<br>"
                + "Time: %{y:.3f} ms<br>"
                + "Shape: %{text}<br>"
                + "<extra></extra>"
            ),
            text=shapes,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=tensor_sizes,
            y=libtorch_times,
            mode="markers",
            name="LibTorch (C++ PyTorch backend)",
            marker=dict(size=8, color="red", symbol="square"),
            hovertemplate=(
                "<b>LibTorch</b><br>"
                + "Tensor Size: %{x}<br>"
                + "Time: %{y:.3f} ms<br>"
                + "Shape: %{text}<br>"
                + "<extra></extra>"
            ),
            text=shapes,
        ),
        row=1,
        col=1,
    )

    # Calculate and plot speedup (inverse of ratio for easier reading)
    speedups = []
    for i, (train_time, libtorch_time) in enumerate(zip(train_times, libtorch_times)):
        if libtorch_time > 0:
            speedup = (
                libtorch_time / train_time
            )  # Speedup = LibTorch time / Train Station time
            speedups.append(speedup)
        else:
            speedups.append(1.0)  # Default to 1.0 if LibTorch time is 0

    # Add speedup plot (pure scatter, no lines)
    fig.add_trace(
        go.Scatter(
            x=tensor_sizes,
            y=speedups,
            mode="markers",
            name="Speedup",
            marker=dict(size=8, color="green", symbol="diamond"),
            hovertemplate=(
                "<b>Speedup</b><br>"
                + "Tensor Size: %{x}<br>"
                + "Speedup: %{y:.3f}x<br>"
                + "Shape: %{text}<br>"
                + "<extra></extra>"
            ),
            text=shapes,
        ),
        row=2,
        col=1,
    )

    # Add reference line at speedup = 1.0
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Equal Performance",
        row=2,
        col=1,
    )

    # Extract hardware information
    hardware_info = extract_hardware_info(data)

    # Create comprehensive title with test statistics and hardware info
    title_text = f"{operation_name} Performance Analysis"
    if avg_iterations > 0:
        title_text += f" ({avg_iterations:,} iterations avg"
        if avg_warmup > 0:
            title_text += f", {avg_warmup} warmup avg"
        title_text += f", {len(data)} tests)"

    # Note: Hardware info will be added as a separate annotation below

    # Update layout with auto-sizing
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=20)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        # Auto-size to fill the screen
        autosize=True,
        height=None,  # Let it auto-size
        width=None,  # Let it auto-size
        # Make it responsive - increased top margin for hardware subtitle and legend
        margin=dict(l=50, r=50, t=150, b=50),
    )

    # Update x-axis for both subplots
    fig.update_xaxes(title_text="Tensor Size (elements)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Tensor Size (elements)", type="log", row=2, col=1)

    # Update y-axes
    fig.update_yaxes(title_text="Time (ms)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Speedup (Train Station / LibTorch)", row=2, col=1)

    # Add annotations for performance insights
    avg_speedup = np.mean(speedups)
    if avg_speedup > 1.0:
        performance_note = f"Train Station is {avg_speedup:.2f}x faster on average"
        color = "green"
    else:
        performance_note = f"LibTorch (C++ PyTorch backend) is {1 / avg_speedup:.2f}x faster on average"
        color = "red"

    fig.add_annotation(
        text=performance_note,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=14, color=color),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=color,
        borderwidth=1,
    )

    # Add hardware information as a separate annotation with smaller font
    if hardware_info:
        fig.add_annotation(
            text=f"Hardware: {hardware_info}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.94,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
        )

    return fig


def create_individual_performance_plot(
    data: List[Dict], plot_type: str, output_file: Optional[str] = None
) -> go.Figure:
    """
    Create an individual performance plot (either timing comparison or speedup).

    Args:
        data: List of performance result dictionaries
        plot_type: Either 'timing' or 'speedup'
        output_file: Optional output file path

    Returns:
        Plotly figure object
    """
    if not data:
        raise ValueError("No performance data provided")

    # Extract operation name for title
    operation_name = extract_operation_name(data)
    hardware_info = extract_hardware_info(data)

    # Calculate test statistics for title
    total_iterations = sum(result.get("iterations", 0) for result in data)
    avg_iterations = total_iterations // len(data) if data else 0
    total_warmup = (
        sum(result.get("warmup_iterations", 0) for result in data)
        if "warmup_iterations" in data[0]
        else 0
    )
    avg_warmup = total_warmup // len(data) if data and total_warmup > 0 else 0

    # Prepare data for plotting
    train_times, libtorch_times, shapes, operations = prepare_plot_data(data)
    tensor_sizes = [calculate_tensor_size(result.get("shape", [])) for result in data]

    if plot_type == "timing":
        # Create timing comparison plot
        fig = go.Figure()

        # Add scatter plots for performance comparison
        fig.add_trace(
            go.Scatter(
                x=tensor_sizes,
                y=train_times,
                mode="markers",
                name="Train Station",
                marker=dict(size=8, color="blue", symbol="circle"),
                hovertemplate=(
                    "<b>Train Station</b><br>"
                    + "Tensor Size: %{x}<br>"
                    + "Time: %{y:.3f} ms<br>"
                    + "Shape: %{text}<br>"
                    + "<extra></extra>"
                ),
                text=shapes,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=tensor_sizes,
                y=libtorch_times,
                mode="markers",
                name="LibTorch (C++ PyTorch backend)",
                marker=dict(size=8, color="red", symbol="square"),
                hovertemplate=(
                    "<b>LibTorch</b><br>"
                    + "Tensor Size: %{x}<br>"
                    + "Time: %{y:.3f} ms<br>"
                    + "Shape: %{text}<br>"
                    + "<extra></extra>"
                ),
                text=shapes,
            )
        )

        # Update layout
        title_text = f"{operation_name} Performance Comparison"
        if avg_iterations > 0:
            title_text += f" ({avg_iterations:,} iterations avg"
            if avg_warmup > 0:
                title_text += f", {avg_warmup} warmup avg"
            title_text += f", {len(data)} tests)"

        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=20)),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            autosize=True,
            height=None,
            width=None,
            margin=dict(l=50, r=50, t=120, b=50),
        )

        fig.update_xaxes(title_text="Tensor Size (elements)", type="log")
        fig.update_yaxes(title_text="Time (ms)", type="log")

    elif plot_type == "speedup":
        # Create speedup plot
        fig = go.Figure()

        # Calculate speedup
        speedups = []
        for train_time, libtorch_time in zip(train_times, libtorch_times):
            if libtorch_time > 0:
                speedup = libtorch_time / train_time
                speedups.append(speedup)
            else:
                speedups.append(1.0)

        fig.add_trace(
            go.Scatter(
                x=tensor_sizes,
                y=speedups,
                mode="markers",
                name="Speedup",
                marker=dict(size=8, color="green", symbol="diamond"),
                hovertemplate=(
                    "<b>Speedup</b><br>"
                    + "Tensor Size: %{x}<br>"
                    + "Speedup: %{y:.3f}x<br>"
                    + "Shape: %{text}<br>"
                    + "<extra></extra>"
                ),
                text=shapes,
            )
        )

        # Add reference line at speedup = 1.0
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Equal Performance",
        )

        # Update layout
        title_text = f"{operation_name} Speedup Analysis"
        if avg_iterations > 0:
            title_text += f" ({avg_iterations:,} iterations avg"
            if avg_warmup > 0:
                title_text += f", {avg_warmup} warmup avg"
            title_text += f", {len(data)} tests)"

        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=20)),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            autosize=True,
            height=None,
            width=None,
            margin=dict(l=50, r=50, t=120, b=50),
        )

        fig.update_xaxes(title_text="Tensor Size (elements)", type="log")
        fig.update_yaxes(title_text="Speedup (Train Station / LibTorch)")

        # Add performance note
        avg_speedup = np.mean(speedups)
        if avg_speedup > 1.0:
            performance_note = f"Train Station is {avg_speedup:.2f}x faster on average"
            color = "green"
        else:
            performance_note = f"LibTorch (C++ PyTorch backend) is {1 / avg_speedup:.2f}x faster on average"
            color = "red"

        fig.add_annotation(
            text=performance_note,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=14, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1,
        )

    # Add hardware information annotation
    if hardware_info:
        fig.add_annotation(
            text=f"Hardware: {hardware_info}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.94,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
        )

    return fig


def main():
    """Main function to run the performance visualization."""
    if len(sys.argv) < 2:
        print(
            "Usage: python performance_scatter.py <json_file> [output_file] [--split]"
        )
        print(
            "Example: python performance_scatter.py add_performance.json add_plot.html"
        )
        print("Example: python performance_scatter.py add_performance.json --split")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = None
    split_plots = False

    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--split":
            split_plots = True
        elif not arg.startswith("--"):
            output_file = arg

    try:
        # Load performance data
        print(f"Loading performance data from: {json_file}")
        data = load_performance_data(json_file)

        if not data:
            print("No performance data found in the JSON file.")
            sys.exit(1)

        # Debug: Show available fields in the first result
        print("Available fields in performance data:")
        for key in data[0].keys():
            print(f"  - {key}: {type(data[0][key]).__name__}")
        print()

        print(f"Loaded {len(data)} performance results")

        # Generate base filename for outputs
        operation_name = extract_operation_name(data).lower().replace(" ", "_")

        if split_plots:
            # Create individual plots
            print("Generating individual performance plots...")

            # Create timing comparison plot
            timing_fig = create_individual_performance_plot(data, "timing")
            timing_output = f"{operation_name}_timing_comparison.html"
            print(f"Saving timing comparison plot to: {timing_output}")
            timing_fig.write_html(timing_output)

            # Create speedup plot
            speedup_fig = create_individual_performance_plot(data, "speedup")
            speedup_output = f"{operation_name}_speedup_analysis.html"
            print(f"Saving speedup analysis plot to: {speedup_output}")
            speedup_fig.write_html(speedup_output)

            print("Individual plots saved successfully:")
            print(f"  - {timing_output}")
            print(f"  - {speedup_output}")
        else:
            # Create combined plot
            print("Generating combined performance scatter plot...")
            fig = create_performance_scatter_plot(data, output_file)

            # Save the plot
            if output_file:
                print(f"Saving plot to: {output_file}")
                fig.write_html(output_file)
                print(f"Plot saved successfully to {output_file}")
            else:
                # Generate default filename
                default_output = f"{operation_name}_performance_plot.html"
                print(f"Saving plot to: {default_output}")
                fig.write_html(default_output)
                print(f"Plot saved successfully to {default_output}")

        # Print summary statistics
        print("\nPerformance Summary:")
        print("===================")

        train_times, libtorch_times, _, _ = prepare_plot_data(data)
        speedups = [
            l / t if t > 0 else 1.0 for t, l in zip(train_times, libtorch_times)
        ]

        avg_train_time = np.mean(train_times)
        avg_libtorch_time = np.mean(libtorch_times)
        avg_speedup = np.mean(speedups)

        # Calculate test statistics
        total_iterations = sum(result.get("iterations", 0) for result in data)
        avg_iterations = total_iterations // len(data) if data else 0
        total_warmup = (
            sum(result.get("warmup_iterations", 0) for result in data)
            if "warmup_iterations" in data[0]
            else 0
        )
        avg_warmup = total_warmup // len(data) if data and total_warmup > 0 else 0

        print("Test Configuration:")
        print(f"  Total tests: {len(data)}")
        print(f"  Average iterations per test: {avg_iterations:,}")
        if avg_warmup > 0:
            print(f"  Average warmup iterations per test: {avg_warmup}")
        print(f"  Total iterations across all tests: {total_iterations:,}")

        print("\nPerformance Results:")
        print(f"  Average Train Station time: {avg_train_time:.3f} ms")
        print(f"  Average LibTorch time: {avg_libtorch_time:.3f} ms")
        print(f"  Average speedup: {avg_speedup:.3f}x")

        if avg_speedup > 1.0:
            print(f"  Train Station is {avg_speedup:.2f}x faster on average")
        else:
            print(
                f"  LibTorch (C++ PyTorch backend) is {1 / avg_speedup:.2f}x faster on average"
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
