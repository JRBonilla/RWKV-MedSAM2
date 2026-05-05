"""Researcher-friendly command-line interface for DRIPP."""

import argparse
import json
import os
import sys

from . import config


def _csv_path():
    return os.path.normpath(os.path.join(config.BASE_UNPROC, config.CSV_FILENAME))


def _print_error(message):
    print(f"Error: {message}", file=sys.stderr)


def _require_csv():
    csv_path = _csv_path()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find datasets CSV at {csv_path}.\n"
            "Update [paths] base_unproc/csv_filename in dripp.ini, or run "
            "`dripp config init` to create a local config template."
        )
    return csv_path


def _load_metadata():
    from .helpers import load_dataset_metadata

    return load_dataset_metadata(_require_csv())


def _print_nested(title, values):
    print(f"{title}:")
    for key, value in values.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")


def cmd_config_show(_args):
    summary = config.get_config_summary()
    print("DRIPP configuration")
    print("===================")
    loaded = summary.pop("config_files_loaded")
    print("Config files loaded:")
    for path in loaded:
        print(f"  {path}")
    print()
    for section, values in summary.items():
        _print_nested(section, values)
        print()
    return 0


def cmd_config_init(args):
    try:
        path = config.write_default_config(args.path, overwrite=args.force)
    except FileExistsError as exc:
        _print_error(f"{exc}\nUse --force to overwrite it.")
        return 2
    print(f"Wrote DRIPP config template: {path}")
    return 0


def cmd_datasets_list(_args):
    try:
        metadata = _load_metadata()
    except Exception as exc:
        _print_error(str(exc))
        return 2

    if not metadata:
        print(f"No datasets found in {_csv_path()}.")
        return 0

    print(f"Datasets from {_csv_path()}:")
    for name, meta in metadata.items():
        modalities = ", ".join(meta.get("modalities") or ["default"])
        state = "preprocessed" if meta.get("preprocessed") else "not preprocessed"
        print(f"  {name} [{modalities}] - {state}")
    return 0


def _save_tasks(csv_path, global_tasks):
    tasks = {
        task: {
            "classes": sorted(list(info["classes"])),
            "datasets": {
                ds: sorted(list(subs))
                for ds, subs in info["datasets"].items()
            },
        }
        for task, info in global_tasks.items()
    }
    tasks_folder = os.path.join(config.INDEX_DIR, "Tasks")
    os.makedirs(tasks_folder, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    tasks_file = os.path.join(tasks_folder, f"{base}_tasks.json")
    with open(tasks_file, "w", encoding="utf-8") as tf:
        json.dump(tasks, tf, indent=2)
    print(f"Saved task summary: {tasks_file}")


def cmd_index(args):
    try:
        csv_path = _require_csv()
        metadata = _load_metadata()
        from .indexer import GLOBAL_TASKS, index_dataset
    except Exception as exc:
        _print_error(str(exc))
        return 2

    if args.dataset:
        if args.dataset not in metadata:
            _print_error(f"Dataset {args.dataset!r} was not found in {csv_path}.")
            return 2
        index_dataset(args.dataset, metadata[args.dataset])
        print(f"Indexed dataset: {args.dataset}")
        return 0

    for name, meta in metadata.items():
        try:
            index_dataset(name, meta)
        except Exception as exc:
            _print_error(f"Error indexing {name!r}: {exc}")
    _save_tasks(csv_path, GLOBAL_TASKS)
    print("Indexing complete.")
    return 0


def cmd_preprocess(args):
    if args.start_at and not args.all:
        _print_error("--start-at can only be used with preprocess --all.")
        return 2

    try:
        csv_path = _require_csv()
        from .dataset import DatasetManager
    except Exception as exc:
        _print_error(str(exc))
        return 2

    if args.gpu:
        config.GPU_ENABLED = True

    try:
        manager = DatasetManager(csv_path)
        if args.dataset:
            if args.dataset not in manager.metadata:
                _print_error(f"Dataset {args.dataset!r} was not found in {csv_path}.")
                return 2
            manager.process_dataset(args.dataset, max_groups=args.max_groups)
        else:
            manager.process_all(start_at=args.start_at, max_groups=args.max_groups)
        export_path = os.path.join(config.BASE_PROC, "dataset_manager.pkl")
        manager.export(export_path)
    except Exception as exc:
        _print_error(str(exc))
        return 1

    print("Preprocessing complete.")
    return 0


def cmd_debugger(_args):
    try:
        import tkinter as tk
        from .debugger.debugger import PreprocessingDebuggerApp
    except Exception as exc:
        _print_error(f"Could not launch debugger: {exc}")
        return 2

    root = tk.Tk()
    PreprocessingDebuggerApp(root)
    root.mainloop()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="dripp",
        description="DRIPP dataset indexing and preprocessing tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Inspect or create DRIPP config files.")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)

    show_parser = config_sub.add_parser("show", help="Show the active DRIPP configuration.")
    show_parser.set_defaults(func=cmd_config_show)

    init_parser = config_sub.add_parser("init", help="Write a commented dripp.ini template.")
    init_parser.add_argument("--path", default="dripp.ini", help="Destination config path.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    init_parser.set_defaults(func=cmd_config_init)

    datasets_parser = subparsers.add_parser("datasets", help="Inspect configured datasets.")
    datasets_sub = datasets_parser.add_subparsers(dest="datasets_command", required=True)
    list_parser = datasets_sub.add_parser("list", help="List datasets from the configured CSV.")
    list_parser.set_defaults(func=cmd_datasets_list)

    index_parser = subparsers.add_parser("index", help="Index one dataset or all datasets.")
    index_group = index_parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("--dataset", help="Dataset name to index.")
    index_group.add_argument("--all", action="store_true", help="Index all datasets.")
    index_parser.set_defaults(func=cmd_index)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess one dataset or all datasets.")
    preprocess_group = preprocess_parser.add_mutually_exclusive_group(required=True)
    preprocess_group.add_argument("--dataset", help="Dataset name to preprocess.")
    preprocess_group.add_argument("--all", action="store_true", help="Preprocess all datasets.")
    preprocess_parser.add_argument("--start-at", help="Dataset name to resume from when using --all.")
    preprocess_parser.add_argument("--max-groups", type=int, default=0, help="Maximum groups per split; 0 means no limit.")
    preprocess_parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for this run.")
    preprocess_parser.set_defaults(func=cmd_preprocess)

    debugger_parser = subparsers.add_parser("debugger", help="Launch the Tk debugger.")
    debugger_parser.set_defaults(func=cmd_debugger)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
