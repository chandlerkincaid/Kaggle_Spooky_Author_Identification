import com.typesafe.config.ConfigFactory
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._

object main {
  def main(args: Array[String]) = {

    /*timer*/
    val StartTime = System.nanoTime
    println("Starting Program")
    /*load spark conf*/
    val SparkConf = new SparkConf().setAppName("main")
    val sc = new SparkContext(SparkConf)
    sc.setLogLevel("WARN")
    /*workaround for toDF import*/
    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    /*load configure tool*/
    val conf = ConfigFactory.load()
    /*load spark session*/
    val spark = SparkSession.builder.master("local").appName("Spooky").getOrCreate()

    /*load data*/
    val train = sc.textFile(conf.getString("data.train")).map(_.split("\",\"").map(_.replaceAll("[^\\w\\s]", "")))
        .map(a => (a(0), a(1), a(2))).toDF("id", "passage", "author")
    val test = sc.textFile(conf.getString("data.test")).map(_.split("\",\"").map(_.replaceAll("[^\\w\\s]", "")))
      .map(a => (a(0), a(1))).toDF("id", "passage")

    val indexer = new StringIndexer()
        .setInputCol("author")
          .setOutputCol("label")

    val tokenizer = new Tokenizer().setInputCol("passage").setOutputCol("words")

    val nGramSettings = Array(1, 2, 3, 4, 5)

    def createNgrammar(inputColumn: String, nValue: Int, vocabSize: Int, minDF: Double): (NGram, CountVectorizer) ={
      (new NGram().setInputCol(inputColumn).setOutputCol(nValue + "-Gram").setN(nValue),
        new CountVectorizer().setInputCol(nValue + "-Gram").setOutputCol(nValue + "-GramVectors")
          .setVocabSize(vocabSize).setMinDF(minDF))
    }

    val nGramTransformers = nGramSettings.map(a => createNgrammar("words", a, 10000, 0.0))

    val Assembler = new VectorAssembler()
      .setInputCols(nGramSettings.map(a => a + "-GramVectors"))
      .setOutputCol("features")

    val classi = new NaiveBayes()
        .setLabelCol("label")
          .setPredictionCol("prediction")
          .setProbabilityCol("probability")

    val trainStages = Array(indexer, tokenizer) ++
      nGramTransformers.map(_._1) ++
      nGramTransformers.map(_._2) ++
      Array(Assembler, classi)

    val trainPipeline = new Pipeline().setStages(trainStages)

    val Array(bTrain, bTest) = train.randomSplit(Array(0.1, 0.1), seed = 1234L)

    val model = trainPipeline.fit(bTrain)

    val predictions = model.transform(bTest)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)

    predictions.select("probability").rdd.take(20).foreach(println)

    val duration = (System.nanoTime - StartTime) / 1e9d
    println("Time to Execute: " + duration + " Seconds")
    println("DONE") //debugging

  }
}
