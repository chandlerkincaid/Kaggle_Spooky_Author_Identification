import java.io._
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
//    val train = sc.textFile(conf.getString("data.train")).map(_.split("\",\"").map(_.replaceAll("[^\\w\\s]", "")))
//        .map(a => (a(0), a(1), a(2))).toDF("id", "passage", "author")
//    val test = sc.textFile(conf.getString("data.test")).map(_.split("\",\"").map(_.replaceAll("[^\\w\\s]", "")))
//      .map(a => (a(0), a(1), "EAP")).toDF("id", "passage", "author")

    val train = sc.textFile(conf.getString("train.csv")).map(_.split("\",\"").map(_.replaceAll("[^\\w\\s]", "")))
      .map(a => (a(0), a(1), a(2))).toDF("id", "passage", "author")
    val test = sc.textFile(conf.getString("test.csv")).map(_.split("\",\"").map(_.replaceAll("[^\\w\\s]", "")))
      .map(a => (a(0), a(1), "EAP")).toDF("id", "passage", "author")

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

    val assembler = new VectorAssembler()
      .setInputCols(nGramSettings.map(a => a + "-GramVectors"))
      .setOutputCol("features")

    val classi = new RandomForestClassifier()
        .setLabelCol("label")
          .setPredictionCol("prediction")
          .setProbabilityCol("probability")

//    val stringer = new IndexToString()
//      .setInputCol("prediction")
//      .setOutputCol("predictioncats")

    val trainStages = Array(indexer, tokenizer) ++
      nGramTransformers.map(_._1) ++
      nGramTransformers.map(_._2) ++
      Array(assembler, classi)

    val trainPipeline = new Pipeline().setStages(trainStages)

    //val Array(bTrain, bTest) = train.randomSplit(Array(0.1, 0.1), seed = 1234L)

    val model = trainPipeline.fit(train)

    val predictions = model.transform(test)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    //val accuracy = evaluator.evaluate(predictions)
    //println("Test set accuracy = " + accuracy)

    val outputRDD = predictions.select("id","probability").rdd.map(a =>
      (a(0).asInstanceOf[String],
      a(1).asInstanceOf[org.apache.spark.ml.linalg.DenseVector].toArray.map(_.toString).mkString(",")
      )
    )

    val fileOutput = new File("output.csv")
    val buff = new BufferedWriter(new FileWriter(fileOutput))
    buff.write("id,EAP,HPL,MWS\n")
    outputRDD.collect().foreach(a => buff.write(a._1 + "," + a._2 + "\n"))
    buff.close()

    val duration = (System.nanoTime - StartTime) / 1e9d
    println("Time to Execute: " + duration + " Seconds")
    println("DONE") //debugging

  }
}
