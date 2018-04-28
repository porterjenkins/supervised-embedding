for i in `seq 1 6`
do
	echo "skip_window size:" $i
	python unsupervised_embedding.py --skip_window $i
done
