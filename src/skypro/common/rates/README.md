# Rates and Flows

A microgrid is modelled as having seven fundamental flows of energy as illustrated below:

![flows](../../../../docs/flows.png)

### Volumetric and fixed rates

One of the key aspects of simulating and reporting on microgrids is to model the costs and revenues associated with using and generating power.
This information is modelled as 'rates' in the codebase (see `src/skypro/common/rates/rates.py`).
Rates can be split into these two categories:
- Volumetric rates, which have the unit `p/kWh`, are charged based on how much energy flows
- Fixed cost rates, which have units like `p/day` or `p/kVA/day`, and do not change depending on how much energy flows.

The volumetric rates are of most interest to control strategies as these costs can be influenced by the control strategy.
The fixed costs are not influenced by the control strategy but they are modelled for completeness and for reporting purposes.

### Market and internal rates

When configuring a simulation or reporting run, the rates can either be defined in local YAML configuration files, or they can be pulled form a central rates database.
The rates that are input into a simulation or reporting run are known in the codebase as *market rates*: market rates lead to actual costs and revenues which effect the cashflow of the ESCO, and are settled with third party Suppliers.

In order to run control strategies, there is also a need to model the notional value of energy, even if there is no concrete cashflow associated with it.
For example, the `solar_to_batt` flow will likely have a `0 p/kWh` *market* rate because no Supplier is going to charge us for using our own on-site solar power, and so the cashflow position of the ESCO is not changed by that flow.
However, the codebase associates an *internal* value/rate with that energy flow, for example, the `solar_to_batt` *internal* rate is calculated as the opportunity cost of not exporting that power to grid. 


### On-site Allocation Methodology (OSAM)

The P395 Elexon modification defines the OSAM calculations, which are used to estimate how much energy that was imported from the grid should count as final demand for the purposes of levies.
Without P395/OSAM we pay final consumption levies on all imported energy, even if the energy is later returned back to the grid in battery trading.

The OSAM calculations diverge from our own methodology on two counts:
1. They define a different merit-order for estimating how energy flows between microgrid components
2. They work on half-hourly meter readings, whereas our calculations can be much more granular (our reporting currently runs at 5 minutely granularity).

Our own methodology is seen as more realistic and more generally useful so is used for most decision-making and reporting, 
whilst the OSAM methodology is important for settlement and optimising against UK grid revenues in the specific case of P395.
 
These two methodologies are effectively run in parallel in the codebase, and any discrepancies in revenue/costs between the two are reported as Notices to the user at the end of a run. 

