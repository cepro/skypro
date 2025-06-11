import argparse
import logging
import importlib.metadata

from skypro.commands.report.main import report_cli
from skypro.commands.simulator.main import simulate


DEFAULT_ENV_FILE = '~/.simt/env.json'


def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO)  # Set to logging.INFO for non-debug mode

    version = importlib.metadata.version('skypro')
    logging.info(f"Skypro version {version}")

    # Create a dictionary of commands, mapping to their python function
    commands = {
        "simulate": simulate,
        "report": report_cli
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser')

    parser_simulate = subparsers.add_parser('simulate')
    parser_simulate.add_argument(
        '-c', '--config',
        dest='config_file_path',
        required=True,
        help='YAML configuration file for this simulation'
    )
    parser_simulate.add_argument(
        '--sim',
        dest='chosen_sim_name',
        required=True,
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
    add_env_file_arg(parser_simulate)

    parser_report = subparsers.add_parser('report')
    parser_report.add_argument(
        '-c', '--config',
        dest='config_file_path',
        required=True,
        help='The YAML configuration file for the report'
    )
    parser_report.add_argument(
        '-m', '--month',
        dest='month_str',
        required=True,
        help='The month to report for, e.g. 2024-04'
    )
    parser_report.add_argument(
        '-o', '--output',
        dest='output_file_path',
        default=None,
        help='Optionally specify an output file path to write a CSV to, with half-hour granularity'
    )
    parser_report.add_argument(
        '-s', '--summary-output',
        dest='summary_output_file_path',
        default=None,
        help='Optionally specify an output file path to write a CSV summary to'
    )
    parser_report.add_argument(
        '-p', '--plot',
        dest='do_plots',
        action="store_true",
        help='If specified, plots will be generated and shown in your default browser.'
    )
    parser_report.add_argument(
        '--save-profiles',
        dest='do_save_profiles',
        action="store_true",
        help='If specified, the load and solar profiles will be saved for future use in simulations. They are saved '
             'into the directory given by `profilesSaveDir` in the config file.'
    )
    parser_report.add_argument(
        '-y',
        dest='skip_cli_warnings',
        action="store_true",
        help='If specified, command line warnings will be auto-accepted.'
    )
    add_env_file_arg(parser_report)

    kwargs = vars(parser.parse_args())

    command = kwargs.pop('subparser')
    if command is None:
        parser.print_help()
        exit(-1)

    commands[command](**kwargs)


def add_env_file_arg(parser):
    parser.add_argument(
        '-e', '--env',
        dest='env_file_path',
        default=DEFAULT_ENV_FILE,
        help=f'JSON file containing environment and secret configuration, defaults to {DEFAULT_ENV_FILE}'
    )


if __name__ == "__main__":
    main()
