package com.databricks.apps.twitter_classifier

import java.util.concurrent.atomic.AtomicInteger

import com.google.gson.Gson
import org.apache.spark.streaming.twitter.TwitterUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

import com.datastax.spark.connector._

/**
 * Collect at least the specified number of tweets into json text files.
 */
object Collect {

  private var numTweetsCollected = 0L
  private var partNum = 0
  private var gson = new Gson()
  private var tweetId:AtomicInteger = new AtomicInteger()

  def main(args: Array[String]) {

    def $(s:String) = sys.env(s)
    System.setProperty("twitter4j.oauth.consumerKey", $("TWITTER_CONSUMER_KEY"))
    System.setProperty("twitter4j.oauth.consumerSecret", $("TWITTER_CONSUMER_SECRET"))
    System.setProperty("twitter4j.oauth.accessToken", $("TWITTER_ACCESS_TOKEN"))
    System.setProperty("twitter4j.oauth.accessTokenSecret", $("TWITTER_ACCESS_TOKEN_SECRET"))

    // Process program arguments and set properties
    if (args.length < 2) {
      System.err.println("Usage: " + this.getClass.getSimpleName +
        "<numTweetsToCollect> <intervalInSeconds>")
      System.exit(1)
    }
    val Array(Utils.IntParam(numTweetsToCollect),  Utils.IntParam(intervalSecs)) =
      Utils.parseCommandLineWithTwitterCredentials(args)

    println("Initializing Streaming Spark Context...")
    println(s"numTweetsToCollect = $numTweetsToCollect")

    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, Seconds(intervalSecs))

    val tweetStream = TwitterUtils.createStream(ssc, None)


  //    .map(gson.toJson(_))

    tweetStream.foreachRDD((rdd, time) => {
      val count = rdd.count()
      println(s"count = $count")
      if (count > 0) {
        val tweetPair = rdd.map(tweet => (tweet.getId,tweet.getText))
        tweetPair.saveToCassandra("tweet_db", "raw_tweets", SomeColumns("tweet_id", "raw_tweet"))
        numTweetsCollected += count
       /* if (numTweetsCollected > numTweetsToCollect) {
          System.exit(0)
        }*/
      }
    })

    ssc.start()
    ssc.awaitTermination()
  }
}
