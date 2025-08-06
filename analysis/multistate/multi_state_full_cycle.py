from analysis.multistate.multi_state_discharge import perform_discharge_analysis
from analysis.multistate.multi_state_refuel import perform_refuel_analysis
from analysis.multistate.multi_state_dormancy import perform_dormancy_analysis

def perform_analysis():
    perform_discharge_analysis()
    # perform_refuel_analysis()
    # perform_dormancy_analysis()


if __name__ == "__main__":
    perform_analysis()