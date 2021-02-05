#!/usr/bin/env bash

INPUT=${1}
OUTPUT=${2}

COMMAND=${3}

start=`date +%s`

${COMMAND} '\"subreddit\":\"ejuice_reviews\|electronic_cigarette\|ecr_eu\|Snus\|gearbest\|PipeTobacco\|DIY_eJuice\|Canadian_ecigarette\|juiceswap\|stopsmoking\|StonerProTips\|Cigarettes\|Vaping\|ecig_vendors\|ecigclassifieds\|juul\|vaporents\|VapePorn\|portabledabs\|Waxpen\|ploompax\|Vaping101\|shitty_ecr\|iqos\|vapeitforward\|hookah\|vapenation\|ECR_UK\|EJuicePorn' ${INPUT} | gzip > ${OUTPUT}

end=`date +%s`
runtime=$((end-start))
echo "Process time ${runtime} seconds"