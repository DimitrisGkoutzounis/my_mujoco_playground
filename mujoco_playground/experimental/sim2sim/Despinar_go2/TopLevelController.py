# TopLevelController class: basically...handles everything
from NavigationPolicy import NavigationPolicy
from Locomotion_Controller import Locomotion_Controller
import numpy as np

class TopLevelController:
    def __init__(
            self, 
            navigationPolicy=None,
            locomotionController=None,
            ):
        
        if navigationPolicy is None:
            raise ValueError("navigationPolicy must be provided.")
        else:
            self.navigationPolicy_ = navigationPolicy

        if locomotionController is None:
            raise ValueError("locomotionController must be provided.")
        else:
            self.locomotionController_ = locomotionController


    