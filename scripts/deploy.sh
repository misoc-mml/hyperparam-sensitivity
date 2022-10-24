for i in {1..10}
do
    #echo $1, $2, $i
    tmux new-window -d "bash run_preds_news_param.sh $1 $2 $i"
done