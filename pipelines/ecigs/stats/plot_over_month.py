import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fn_lists = {
    'twitter_relevant': {'Relevant Tweets': 'twitter_relevant_month.csv',
                         'Relevant Tweets in US': 'twitter_relevant_us_month.csv'},
    'reddit_comment': {'Keywords filtered Reddit comments': 'reddit_comment_keywords_month.csv',
                       'Keywords filtered Reddit comments in US': 'reddit_comment_keywords_us_month.csv'},
    'reddit_submission': {'Keywords filtered Reddit submissions': 'reddit_submission_keywords_month.csv',
                          'Keywords filtered Reddit submissions in US': 'reddit_submission_keywords_us_month.csv'}
}

ylabel_mapping = {
    'twitter_relevant': 'Number of Tweets',
    'reddit_comment': 'Number of Reddit comments',
    'reddit_submission': 'Number of Reddit submissions'
}

for out_name, fn_list in fn_lists.items():
    aggregated_df = pd.DataFrame()
    df_list = []
    for col, fn in fn_list.items():
        print(fn)
        df = pd.read_csv(fn)
        # print(df.sum(axis='num_tweets'))
        df.rename({'num_tweets': col, 'num': col}, axis='columns', inplace=True)
        df_list.append(df)

    aggregated_df = df_list[0].merge(df_list[1], how='left', on='month')
    print(aggregated_df.head())

    # plot
    plt.figure(figsize=(16, 8))
    plt.rcParams["font.family"] = "Times New Roman"
    rc = {'axes.labelsize': 20, 'axes.titlesize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
    sns.set(rc=rc)
    sns.set_style('whitegrid')

    # g = sns.relplot(data=aggregated_df.melt(id_vars=['month'], var_name=''), kind='line', height=8, aspect=2,
    #                 x='month', y='value', hue='')
    g = sns.lineplot(data=aggregated_df.melt(id_vars=['month'], var_name=''), x='month', y='value', hue='')
    g.set(xticks=list(range(0, len(aggregated_df), 6)))
    g.set(xlabel='Month', ylabel=ylabel_mapping[out_name])
    # g.fig.autofmt_xdate()
    plt.setp(g.lines, linewidth=5)
    plt.legend(loc='upper left', fontsize=20)
    plt.setp(g.get_xticklabels(), rotation=30, ha='right')

    plt.show()
    save_fig = g.get_figure()
    save_fig.savefig('{}.png'.format(out_name))
