# Skypro reporting web app

This is a Streamlit web app that makes the reporting functionality available over a web front end.

The app supports different 'scenarios'- each of which have an associated reporting configuration file.
This is useful because it allows multiple microgrids to be reviewed from the same app.
It also means you can compare multiple Supplier arrangements, for example, you could have a scenario where P395 is active, and one where it's not.

The app calls into the main reporting code (using the appropriate configuration file) and visualises the results.

![screenshot](../../../docs/reporting_webapp_screenshot.png)

## Usage

The Streamlit app can be launched with:
```
SKIP_PASSWORD=true CONFIG_FILE=src/skypro/reporting_webapp/example_config.yaml streamlit run src/skypro/reporting_webapp/main.py
```
This should open a window in your default browser.
This example configuration uses the integration testing fixtures which only have data available for August 2024 - so that is the only month that works in this example!
 