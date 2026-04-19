import warnings
warnings.filterwarnings("ignore")

from package.app_manager import AppManager

if __name__=="__main__":
    app_manager = AppManager()
    app_manager.main_loop()
