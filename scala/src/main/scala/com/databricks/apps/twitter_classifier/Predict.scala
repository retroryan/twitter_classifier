package com.databricks.apps.twitter_classifier

import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SQLContext
import org.apache.spark.streaming.twitter._
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Pulls live tweets and filters them for tweets in the chosen cluster.
 */
object Predict {
  def main(args: Array[String]) {
    if (args.length < 1) {
      System.err.println("Usage: " + this.getClass.getSimpleName + " <clusterNumber>")
      System.exit(1)
    }

    val clusterNumber = Integer.parseInt(args(0))

    def $(s: String) = sys.env(s)
    System.setProperty("twitter4j.oauth.consumerKey", $("TWITTER_CONSUMER_KEY"))
    System.setProperty("twitter4j.oauth.consumerSecret", $("TWITTER_CONSUMER_SECRET"))
    System.setProperty("twitter4j.oauth.accessToken", $("TWITTER_ACCESS_TOKEN"))
    System.setProperty("twitter4j.oauth.accessTokenSecret", $("TWITTER_ACCESS_TOKEN_SECRET"))

    println("Initializing Streaming Spark Context...")

    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
    val sparkContext = new SparkContext(conf)
    val sqlContext = new SQLContext(sparkContext)
    val ssc = new StreamingContext(sparkContext, Seconds(5))


    //SomeColumns("vector_id","size", "indices","values")

    val clusterVectorsDF = sqlContext.read.format("org.apache.spark.sql.cassandra")
      .options(Map("keyspace" -> "tweet_db", "table" -> "cluster_vectors"))
      .load()

    clusterVectorsDF.show()

    // .toDF("vector_id", "size", "indices", "values")

    println(s"cluster vectors count: ${clusterVectorsDF.count()}")

    val clusterCenters: Array[Vector] = clusterVectorsDF.collect().map(row =>
      Vectors.parse(row.getString(1)))



    println("Initializing Twitter stream...")
    val tweets = TwitterUtils.createStream(ssc, None)
    val statuses = tweets.map(tweet => tweet.getText)

    println("Initalizaing the the KMeans model...")
    //val rdd: RDD[Vector] = ssc.sparkContext.objectFile[Vector](modelFile.toString)

    val model = new KMeansModel(clusterCenters)

    val filteredTweets = statuses
      .filter(t => {
      val predict: Int = model.predict(Utils.featurize(t))
      predict == clusterNumber
    })
    filteredTweets.print()

    // Start the streaming computation
    println("Initialization complete.")
    ssc.start()
    ssc.awaitTermination()
  }
}
