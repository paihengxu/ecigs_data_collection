

def read_txt(fn):
    print(fn)
    result_set = set()
    for line in open(fn, 'r'):
        result_set.add(line.strip().lower())

    print("unique terms:", len(result_set))
    return result_set


def read_csv(fn):
    result_dict = {}
    for idx, line in enumerate(open(fn, 'r')):
        if idx == 0:
            continue
        term, num = line.strip().split(',')
        result_dict[term] = int(num)
    return result_dict


def aggregate_counts(keywords_set, reddit, twitter, output_fn):
    sorted_keywords = sorted(keywords_set)
    with open(output_fn, 'w') as outf:
        outf.write("Keywords,Reddit mention times,Twitter mention times\n")
        for ele in sorted_keywords:
            reddit_num = reddit.get(ele, 0)
            twitter_num = twitter.get(ele, 0)
            outf.write('"{}",{},{}\n'.format(ele, str(reddit_num), str(twitter_num)))


if __name__ == '__main__':
    manual_set = read_txt('../../../python/falconet/resources/annotators/keywords/ecig.keywords')
    manual_refined_brands = read_txt(
        '../../../python/falconet/resources/annotators/keywords/ENDS_refined.keywords')
    manual_all_brands = read_txt(
        '../../../python/falconet/resources/annotators/keywords/ENDS.keywords')
    all_brands = read_txt(
        '../../../python/falconet/resources/annotators/keywords/ecigs_brands.keywords')

    refined_brands = manual_refined_brands - manual_set

    twitter_counts = read_csv('twitter_relevant_keywords_counts.csv')
    reddit_counts = read_csv('reddit_keywords_counts.csv')

    aggregate_counts(manual_set, reddit_counts, twitter_counts, 'keywords_manual.csv')
    aggregate_counts(refined_brands, reddit_counts, twitter_counts, 'keywords_refined_brands.csv')
