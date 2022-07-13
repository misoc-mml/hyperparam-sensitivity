# Options: sl, tl
MODELS=("sl")
BASE_MODELS=("l1" "l2" "tr" "dt" "rf" "et" "kr" "cb" "lgbm")

for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        echo ${MODEL}_${BASE_MODEL}

        zip -r -q ${MODEL}_${BASE_MODEL}.zip ${MODEL}_${BASE_MODEL}
    done
done