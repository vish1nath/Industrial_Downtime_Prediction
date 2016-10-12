if [ $# -eq 0 ]
then
echo "Please specify the Model to run"
exit 1
fi

if [ $1 = "LR" ]
then
echo "Running Logistic Regression"
python Logistic_Regression.py
exit  0
fi

if [ $1 = 'NB' ]
then
echo "Running Naive Bayes"
python Naive_Bayes.py
exit  0
fi

if [ $1 = 'RF' ]
then
echo "Running Random Forest"
python Random_Forest.py
exit  0
fi

if [ $1 = 'GB' ] 
then
echo "Running Gradient Boosting"
python Gradient_Boosting.py
exit  0
fi

echo "please choose between LR,NB,RF and GB"
exit 1
