
from enum import Enum

class Params(Enum):
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

