import argparse
from skypro.commands import prepare, validate

def main():
    parser = argparse.ArgumentParser(prog="skypro", description="Skyprospector data generator")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Add sub-command parsers
    prepare_parser = subparsers.add_parser("prepare", help="Run prepare")
    validate_parser = subparsers.add_parser("validate", help="Run validate")
    validate_parser.add_argument("folder", help="Path to the folder to validate")

    #parser.add_argument("--name", help="Your name")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare.main()
    elif args.command == "validate":
        validate.main(args.folder)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
