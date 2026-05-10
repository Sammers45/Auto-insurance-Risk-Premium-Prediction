from pathlib import Path
import sys

sys.path.append(str(Path('src').resolve()))
from data_preprocessing import create_data_if_missing

source = Path('..') / 'insurance.csv'
output = Path('data') / 'insurance_data.csv'

create_data_if_missing(source, output)
print('Generated dataset at', output.resolve())
