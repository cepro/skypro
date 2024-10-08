import argparse
import logging
import importlib.metadata

from skypro.commands.simulator.main import simulate


DEFAULT_ENV_FILE = "~/.simt/env.json"


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
        '--sim',
        dest='chosen_sim_name',
        default=None,
        help='When using a V4 configuration file, this is the name of the simulation case to run. Or "all" to run every'
             ' simulation.'
    )
    parser_simulate.add_argument(
        '-p', '--plot',
        dest='do_plots',
        action="store_true",
        help='If specified, plots will be generated and shown in your default browser.'
    )
    parser_simulate.add_argument(
        '-y',
        dest='skip_cli_warnings',
        action="store_true",
        help='If specified, command line warnings will be auto-accepted.'
    )

    kwargs = vars(parser.parse_args())

    command = kwargs.pop('subparser')
    if command is None:
        parser.print_help()
        exit(-1)

    commands[command](**kwargs)


if __name__ == "__main__":
    main()
