import warnings

from package.app_manager import AppManager
from saved_state_manager import SavedStateManager

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    saved_state_manager = SavedStateManager()
    app_manager = AppManager(saved_state_manager)
    app_manager.main_loop()
