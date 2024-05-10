import argparse
import logging
import importlib.metadata

from skypro.commands.simulator.main import simulate


DEFAULT_ENV_FILE = f"~/.simt/env.json"


def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO)  # Set to logging.INFO for non-debug mode

    version = importlib.metadata.version('skypro')
    logging.info(f"Skypro version {version}")

    # Create a dictionary of commands, mapping to their python function
    commands = {
        "simulate": simulate
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser')

    parser_simulate = subparsers.add_parser('simulate')
    parser_simulate.add_argument(
        '-c', '--config',
        dest='config_file_path',
        required=True,
        help='JSON configuration file for this simulation'
    )
    parser_simulate.add_argument(
        '-e', '--env',
        dest='env_file_path',
        default=DEFAULT_ENV_FILE,
        help=f'JSON file containing environment and secret configuration, defaults to {DEFAULT_ENV_FILE}'
    )
    parser_simulate.add_argument(
        '-o', '--output',
        dest='output_file_path',
        default=None,
        help='Output CSV file location, by default no output file is generated.'
    )
    parser_simulate.add_argument(
        '-p', '--plot',
        dest='do_plots',
        action="store_true",
        help='If specified, plots will be generated and shown in your default browser.'
    )

    kwargs = vars(parser.parse_args())

    command = kwargs.pop('subparser')
    if command is None:
        parser.print_help()
        exit(-1)

    commands[command](**kwargs)


if __name__ == "__main__":
    main()
