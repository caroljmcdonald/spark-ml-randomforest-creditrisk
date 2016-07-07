
This is an example using Spark Machine Learning Random Forests , written in Scala, 


There is  1 datafile  in this directory :
	germancredit.csv  
 
You will need to copy this files to your MapR sandbox, or wherever you have Spark installed.

You can run these examples in the spark shell by putting the code from the scala file in the spark shell after launching:
 
$spark-shell --master local[1]

Or you can run the applications with these steps:

Step 1: First compile the project: Select project  -> Run As -> Maven Install

Step 2: Copy the spark-ml-randomforest-1.0.jar to the sandbox 

To run the  standalone :

spark-submit --class example.credit --master yarn spark-ml-randomforest-1.0.jar


