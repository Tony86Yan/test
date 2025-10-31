# -*- coding: utf-8 -*-
import io
import sys

print("????UTF-8?????...")

# ????????????????????
with io.open('timeseries_trend_analysis_extended_utf8.py', 'w', encoding='utf-8') as f:
    f.write('# -*- coding: utf-8 -*-\n')
    f.write('"""\n')
    f.write('?????????????????\n')
    f.write('??????????????????????/??/????????\n')
    f.write('???????????????\n')
    f.write('"""\n\n')
    f.write('import numpy as np\n')
    f.write('import pandas as pd\n')
    f.write('from scipy import stats\n')
    f.write('from sklearn.linear_model import LinearRegression\n')
    f.write('from sklearn.preprocessing import PolynomialFeatures\n')
    f.write('from sklearn.metrics import r2_score\n')
    f.write('from datetime import datetime, timedelta\n')
    f.write('import warnings\n')
    f.write('import logging\n')
    f.write('import sys\n')
    f.write('import os\n\n')
    f.write('# ???????UTF-8\n')
    f.write('if sys.version_info >= (3, 7):\n')
    f.write('    if hasattr(sys.stdout, "reconfigure"):\n')
    f.write('        sys.stdout.reconfigure(encoding="utf-8")\n')
    f.write('    if hasattr(sys.stderr, "reconfigure"):\n')
    f.write('        sys.stderr.reconfigure(encoding="utf-8")\n\n')
    f.write('warnings.filterwarnings("ignore")\n\n\n')
    f.write('logger = logging.getLogger()\n')
    f.write('logging.basicConfig(\n')
    f.write('    filename="./logs/timeseries_trend_analysis.log",\n')
    f.write('    filemode="a",\n')
    f.write('    level=logging.INFO,\n')
    f.write('    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",\n')
    f.write('    datefmt="%Y-%m-%d %H:%M:%S",\n')
    f.write('    encoding="utf-8"\n')
    f.write(')\n\n\n')

print("????????")

# ????
with io.open('timeseries_trend_analysis_extended_utf8.py', 'r', encoding='utf-8') as f:
    content = f.read(300)
    print("\n???300????")
    print(content)
    print("\n????????UTF-8???!")
