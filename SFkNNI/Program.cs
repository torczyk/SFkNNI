using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;

namespace kNN
{
    public class ValuePair<T1, T2>
    {
        // value pair
        public T1 Item1 { get; set; }
        public T2 Item2 { get; set; }
        public ValuePair() { }
        public ValuePair(T1 Item1, T2 Item2)
        {
            this.Item1 = Item1;
            this.Item2 = Item2;
        }
    }

    class Program
    {
        public static double SFkNN(ref List<Tuple<double?[], double>> training, Tuple<double?[], double> test, int k, bool useRangeCheck, bool weightedAvg)
        {
            // Local variables declarations
            // nearest and farthest distance
            double d0, dk;
            // ...analyzed neighbors count
            int neiCount;
            // ...valid features count
            int validFeatures = 0;
            // ...valid values count
            int validValues;
            // ...table of mean target values from neighbors for each feature
            double[] targetValues = new double[test.Item1.GetLength(0)];
            // ...table of mean target value weights
            double[] featureWeights = new double[test.Item1.GetLength(0)];
            // ...median ald low/high threshold values
            double medVal = 0;
            double lowLimit = 0;
            double uppLimit = 0;
            bool inLimits;
            int medIdx, idxRange;
            // ...list of attribute values
            List<double> attValues = new List<double>();
            // ...distance table
            List<ValuePair<double, double>> distVector = new List<ValuePair<double, double>>();
            // Loop over features
            for (int feature = 0; feature < test.Item1.GetLength(0); feature++)
            {
                validValues = 0;
                featureWeights[feature] = 0;
                targetValues[feature] = 0;
                // If calssified attribute has a value...
                if (test.Item1[feature].HasValue)
                {
                    inLimits = true;
                    if (useRangeCheck)
                    {
                        // Determining ranges of atrribute
                        inLimits = false;
                        attValues.Clear();
                        for (int row = 0; row < training.Count; row++)
                            if (training[row].Item1[feature].HasValue)
                                attValues.Add(training[row].Item1[feature].Value);
                        attValues.Sort();
                        medIdx = attValues.Count / 2;
                        idxRange = (int)(medIdx * 0.75); // Value range - between 1 and 3 quartile
                        medVal = attValues[medIdx]; // Median value
                        lowLimit = attValues[medIdx - idxRange]; // Min value
                        uppLimit = attValues[medIdx + idxRange]; // Max value
                        if (test.Item1[feature].Value >= lowLimit && test.Item1[feature].Value <= uppLimit)
                            inLimits = true; // Value is within limits
                    }
                    // If classified value is within the acceptable range
                    // for each class in the reference set...
                    if (inLimits)
                    {
                        // Count not empty features of classified object
                        validFeatures++;
                        // Determine distance from classified vector
                        distVector.Clear();
                        for (int row = 0; row < training.Count; row++)
                            if (training[row].Item1[feature].HasValue)
                            {
                                validValues++;
                                featureWeights[feature]++;
                                distVector.Add(new ValuePair<double, double>(training[row].Item2, Math.Abs((double)training[row].Item1[feature] - (double)test.Item1[feature])));
                            }
                        featureWeights[feature] /= (double)training.Count;
                        // Sort result depending on the distance form the classified vector
                        distVector = distVector.OrderBy(x => x.Item2).ToList();
                        // Solve ties on eqidistant neighbors - expand the analysis window "k"
                        for (neiCount = 0; neiCount < k; )
                        {
                            // Determiine how many among k nearest neighbors are equidistant
                            neiCount += distVector.Count(x => x.Item2.Equals(distVector.ElementAt(neiCount).Item2));
                        }
                        // Determine nearest and farthest neigbor among k nearest neighbors
                        d0 = distVector[0].Item2;
                        // dk = distVector[neiCount - 1].Item2;
                        dk = distVector[distVector.Count - 1].Item2;
                        // w = (dMax - d)/(dMax - dMin)
                        // Calculate sum of target values
                        targetValues[feature] = (distVector.Take(neiCount).Sum(x => x.Item1)) / (double)neiCount;
                    }
                }
            }
            // Calculate new target value
            if (weightedAvg)
            {
                double result = 0;
                for (int f = 0; f < test.Item1.GetLength(0); f++)
                {
                    result += targetValues[f] * featureWeights[f];
                }
                return result / featureWeights.Sum();
            }
            else
            {
                return targetValues.Sum() / validFeatures;
            }
        }

        public static string vectorToString(Tuple<double?[], string> vector)
        {
            string result = "";
            foreach (double? value in vector.Item1)
            {
                result += value.ToString() + ",";
            }
            result += vector.Item2.ToString();
            return result;
        }

        public static void parseCSV(string fname, out List<Tuple<double?[], double>> data, out string[] header)
        {
            data = new List<Tuple<double?[], double>>();
            const char delim = ',';
            string line;
            string[] parts;
            int fieldCnt;
            double?[] attribs;
            double targetVal;
            using (StreamReader reader = new StreamReader(fname))
            {
                // Reading the header - determine column cout
                line = reader.ReadLine();
                parts = line.Split(delim);
                fieldCnt = parts.Length;
                // Extracting the header
                header = new string[fieldCnt];
                parts.CopyTo(header, 0);
                // Parse file data into list
                while (true)
                {
                    line = reader.ReadLine();
                    if (line == null)
                    {
                        break;
                    }
                    parts = line.Split(delim);
                    attribs = new double?[fieldCnt - 1];
                    for (int i = 0; i < fieldCnt - 1; i++)
                    {
                        if (parts[i].Equals(""))
                            attribs[i] = null;
                        else
                            attribs[i] = double.Parse(parts[i]);
                    }
                    targetVal = double.Parse(parts[fieldCnt - 1]);
                    data.Add(new Tuple<double?[], double>(attribs, targetVal));
                }
            }
        }

        public static void parseCSV(string fname, out List<Tuple<double?[], string>> data, out string[] header)
        {
            data = new List<Tuple<double?[], string>>();
            const char delim = ',';
            string line;
            string[] parts;
            int fieldCnt;
            double?[] attribs;
            string classLbl;
            using (StreamReader reader = new StreamReader(fname))
            {
                // Reading the header - determine column cout
                line = reader.ReadLine();
                parts = line.Split(delim);
                fieldCnt = parts.Length;
                // Extracting the header
                header = new string[fieldCnt];
                parts.CopyTo(header, 0);
                // Parse file data into list
                while (true)
                {
                    line = reader.ReadLine();
                    if (line == null)
                    {
                        break;
                    }
                    parts = line.Split(delim);
                    attribs = new double?[fieldCnt - 1];
                    for (int i = 0; i < fieldCnt - 1; i++)
                    {
                        if (parts[i].Equals(""))
                            attribs[i] = null;
                        else
                            attribs[i] = double.Parse(parts[i]);
                    }
                    classLbl = parts[fieldCnt - 1].Replace("\"", string.Empty);
                    data.Add(new Tuple<double?[], string>(attribs, classLbl));
                }
            }
        }

        public static void saveCSV(string fname, ref List<Tuple<double?[], string>> fset, ref string[] header)
        {
            Tuple<double?[], string> vec;
            string line;
            using (StreamWriter writer = new StreamWriter(fname))
            {
                // Save header into output file
                writer.WriteLine(string.Join(",", header));
                for (int row = 0; row < fset.Count; row++)
                {
                    vec = fset[row];
                    line = vectorToString(vec);
                    writer.WriteLine(line);
                }
            }
        }

        public static void fillMissing(ref List<Tuple<double?[], string>> mset, out List<Tuple<double?[], string>> fset, int k, bool useRangeCheck, bool targetTypeCheck, bool weightedAvg)
        {
            List<Tuple<double?[], double>> temp;
            Tuple<double?[], double> ttemp;
            Tuple<double?[], string> ftemp;
            bool[] isInteger = new bool[mset[0].Item1.Length]; // false by default (means double)
            int jjj;
            fset = new List<Tuple<double?[], string>>();
            if (targetTypeCheck)
            {
                for (int col = 0; col < isInteger.Length; col++) // loop over attributes (columns) of "M" set - data type detection
                {
                    isInteger[col] = true; // initialize with true (means int)
                    for (int row = 0; row < mset.Count; row++) // loop over rows within the analyzed attribute
                    {
                        if (mset[row].Item1[col].HasValue)
                        {
                            isInteger[col] = (mset[row].Item1[col] == Math.Round(mset[row].Item1[col].Value));
                            if (!isInteger[col]) break; // after first mismatch leave the loop with false (means double)
                        }
                    }
                }
            }
            for (int i = 0; i < mset.Count; i++) // loop over records (rows) of "M" set
            {
                ftemp = new Tuple<double?[], string>(new double?[mset[i].Item1.Length], mset[i].Item2);
                for (int j = 0; j < mset[i].Item1.Length; j++) // loop over attributes within the analyzed record
                {
                    if (mset[i].Item1[j] == null) // if is NULL - start filling missing value
                    {
                        temp = new List<Tuple<double?[], double>>();
                        for (int ii = 0; ii < mset.Count; ii++)
                        {
                            if (ii != i && mset[ii].Item1[j] != null) // if target value is different from NULL
                            {
                                ttemp = new Tuple<double?[], double>(new double?[mset[i].Item1.Length - 1], (double)mset[ii].Item1[j]);
                                jjj = 0;
                                for (int jj = 0; jj < mset[i].Item1.Length; jj++)
                                {
                                    if (jj != j)
                                    {
                                        ttemp.Item1[jjj++] = mset[ii].Item1[jj];
                                    }
                                }
                                temp.Add(ttemp);
                            }
                        }
                        ttemp = new Tuple<double?[], double>(new double?[mset[i].Item1.Length - 1], -1);
                        jjj = 0;
                        for (int jj = 0; jj < mset[i].Item1.Length; jj++)
                        {
                            if (jj != j)
                            {
                                ttemp.Item1[jjj++] = mset[i].Item1[jj];
                            }
                        }
                        ftemp.Item1[j] = SFkNN(ref temp, ttemp, k, useRangeCheck, weightedAvg);
                        if (isInteger[j]) ftemp.Item1[j] = Math.Round(ftemp.Item1[j].Value);
                    }
                    else // it is not NULL - leave as is
                    {
                        ftemp.Item1[j] = mset[i].Item1[j];
                    }
                }
                fset.Add(ftemp);
            }
        }

        // knn.exe <k> <RangeCheck> <TargetTypeCheck>
        // k - minimal number of analyzed neighbors (1 - 99+)
        // RangeCheck - reject values from 4th quartile (t/f)
        // TargetTypeCheck - check if imputed feature is integer type or always impute double value (t/f)

        public static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US", false);

            List<Tuple<double?[], string>> mset;
            List<Tuple<double?[], string>> fset;
            string[] header;

            const string misfile = @"mdata.csv";
            const string impfile = @"fdata.csv";

            int k = int.Parse(args[0]);
            bool rangeCheck = (args[1] == "t" ? true : false);
            bool targetTypeCheck = (args[2] == "t" ? true : false);
            bool weightedAvg = (args[3] == "t" ? true : false);

            parseCSV(misfile, out mset, out header);

            fillMissing(ref mset, out fset, k, rangeCheck, targetTypeCheck, weightedAvg);

            saveCSV(impfile, ref fset, ref header);

            //Console.ReadKey();
        }
    }
}