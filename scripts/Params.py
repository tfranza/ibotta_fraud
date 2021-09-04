
from enum import Enum

class Params(Enum):
    '''
    Stores all parameters and other otherwise-hardcoded variables into one single class.
    
    '''

    #--- Paths
    OUTPUT_FILEPATH = './data/data_analysis/crafted/'
    DATASET_FILEPATH = './data/data_analysis/'


    #--- Translation values from Czech to English
    TYPES = {
        'PRIJEM': 'Credit', 
        'VYDAJ': 'Withdrawal', 
        'VYBER': 'Withdrawal in cash'
    }
    OPERATIONS = {
        'VKLAD': 'Credit in cash', 
        'PREVOD Z UCTU': 'Collection from another bank', 
        'PREVOD NA UCET': 'Remittance to another bank',
        'VYBER': 'Withdrawal in cash',
        'VYBER KARTOU': 'Credit Card Withdrawal'
    }
    KSYMBOLS = {
        'SIPO': 'Household Payment', 
        'SLUZBY': 'Payment of Statement', 
        'UVER': 'Loan Payment',
        'POJISTNE': 'Insurance Payment',
        'DUCHOD': 'Old-age Pension Payment',
        'UROK': 'Interest Credited',
        'SANKC. UROK': 'Sanction Interest'
    }


    #--- Other options
    ROW_SLICE_SIZE = 1000
    CONTAMINATION = 0.03
    RANDOM_STATE = 0

