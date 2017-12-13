import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

// Load and parse the data
val data = sc.textFile("data/mllib/kmeans_data.txt")
val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

// Cluster the data into two classes using KMeans
val numClusters = 2
val numIterations = 20
val clusters = KMeans.train(parsedData, numClusters, numIterations)

// Evaluate Clustering by computing within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println("wintin Set Sum of Squared Errors = " + WSSSE)

// Save and load model 
clusters.save(sc, "myMOdelPath")
val sameModel = KMeansModel.load(sc, "myModelPath")

