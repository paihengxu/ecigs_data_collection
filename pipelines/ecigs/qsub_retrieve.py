import os
from datetime import datetime, timedelta

# CONST
START_DATE = datetime(2019, 9, 1)
END_DATE = datetime(2020, 10, 30)

cmd = "qsub -N retrieve-{message}-{Y}-{m} -cwd -j y -o {log_dir}/{Y}-{m}.log " \
      "-l mem_free=15G,ram_free=15G " \
      "run_retrieve.sh {message} {start_date} {end_date} {out_dir}"

# Please retrieve submission first
message_type = 'submission'
# message_type = 'comment'
out_dir = ''  # TODO


def months_in_range(start_date, end_date):
    """Get the last day of every month in a range between two datetime values.
    Return a generator
    """
    start_month = start_date.month
    end_months = (end_date.year-start_date.year)*12 + end_date.month + 1

    for month in range(start_month+1, end_months+1):
        # Get years in the date range, add it to the start date's year
        year = int((month-1)/12) + start_date.year
        month = (month-1) % 12 + 1

        yield datetime(year, month, 1)-timedelta(days=1)


log_dir = os.path.join(out_dir, 'reddit_{}'.format(message_type), 'retrieve_log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

for date in months_in_range(START_DATE, END_DATE):
    month_end = str(date.date())
    y, m, _ = month_end.split('-')
    month_start = "{}-{}-01".format(y, m)

    print(month_start, month_end)
    os.system(cmd.format(Y=y, m=m, log_dir=log_dir, message=message_type, start_date=month_start,
                         end_date=month_end, out_dir=out_dir))
