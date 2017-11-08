name := "KaggleSpookyAuthor"

version := "1.0"

scalaVersion := "2.11.7"

logLevel := Level.Error

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0",
  "com.typesafe" % "config" % "1.3.1"
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}