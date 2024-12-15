using ScopeRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxWrapper;

public class Utils
{
    public static int float2int(float score)
    {
        if (float.IsNaN(score) || score < 0 || score > 1)
        {
            return (int)-1;
        }
        return (int)(score * 255);
    }

    public static float[] BinaryToFloatArray(byte[] bytes)
    {
        if (bytes == null) return null;
        float[] values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }


    public static byte[] FloatArray2Bytes(float[] values)
    {
        byte[] array = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, array, 0, array.Length);
        return array;
    }
    public static long GetTimestamp()
    {
        long timeStamp = DateTimeOffset.Now.ToUnixTimeSeconds();
        return timeStamp;
    }
    public static bool IsTimestampExpired(long timestamp)
    {
        long curTimeStamp = DateTimeOffset.Now.ToUnixTimeSeconds();
        long duration = 2592000; //30 * 24 * 60 * 60
        long diff = curTimeStamp - timestamp;
        return diff >= duration;
    }
    public static string convertToString(float[] values)
    {
        if (values == null) return "";
        string result = "";
        foreach (var val in values)
        {
            result += Convert.ToString(val) + ",";
        }
        return result.TrimEnd(',');
    }
    public static string convertTimeStampToString(long timeStamp)
    {
        System.DateTime startTime = TimeZone.CurrentTimeZone.ToLocalTime(new System.DateTime(1970, 1, 1));
        DateTime dt = startTime.AddSeconds(timeStamp);
        return dt.ToString("yyyy/MM/dd HH:mm:ss:ffff");
    }

}


public class PrismyV4Row
{
    public string ImageKey { get; set; }
    public float PrismyV4 { get; set; }
    public float EmotionalImpact { get; set; }
    public float Inspiring { get; set; }
    public float Interestingness { get; set; }
    public float Aesthetic { get; set; }
    public float PinStyle { get; set; }

    public PrismyV4Row(string imageKey, float prismyV4, float emotionalImpact, float inspiring, float interestingness, float aesthetic, float pinStyle)
    {
        ImageKey = imageKey;
        PrismyV4 = prismyV4;
        EmotionalImpact = emotionalImpact;
        Inspiring = inspiring;
        Interestingness = interestingness;
        Aesthetic = aesthetic;
        PinStyle = pinStyle;
    }
}
public class PrismyV4ScopeProcessor : Processor
{
    int batchSize = 64;
    private OnnxWrapper.OnnxRunner argusRunner = null;
    private string embeddingPath = string.Empty;
    private int embedSize = 1024;
    private float max_val = 10.0f;
    private float min_val = 0.0f;

    private float TrimScore(float value)
    {
        return (Math.Max(min_val, Math.Min(max_val, value))) / 10.0f;
    }


    public override bool RunSingleThreaded { get { return true; } }
    public override Schema Produces(string[] columns, string[] args, Schema input)
    {
        Schema schema = new Schema();
        schema.Add(new ColumnInfo("ImageKey", ColumnDataType.String));
        schema.Add(new ColumnInfo("PrismyV4", ColumnDataType.Float));
        schema.Add(new ColumnInfo("EmotionalImpact", ColumnDataType.Float));
        schema.Add(new ColumnInfo("Inspiring", ColumnDataType.Float));
        schema.Add(new ColumnInfo("Interestingness", ColumnDataType.Float));
        schema.Add(new ColumnInfo("Aesthetic", ColumnDataType.Float));
        schema.Add(new ColumnInfo("PinStyle", ColumnDataType.Float));
        //        schema.Add(new ColumnInfo("Brightness", ColumnDataType.Float));
        //        schema.Add(new ColumnInfo("Style", ColumnDataType.Float));
        schema.Add(new ColumnInfo("TimeStamp", ColumnDataType.Long));
        return schema;
    }

    private void Init(string[] args)
    {
        if (argusRunner != null) return;
        var modelName = args[0];
        embedSize = int.Parse(args[1]);
        argusRunner = new OnnxWrapper.OnnxRunner(modelName);
    }

    private byte[] FloatArray2Bytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private float[] Bytes2FloatArray(byte[] data)
    {
        float[] values = new float[data.Length / sizeof(float)];
        Buffer.BlockCopy(data, 0, values, 0, data.Length);
        return values;
    }

    //Tuple<string, float, float, float, float, float, float, float, float>

    private List<PrismyV4Row> OutputRows(List<string> keys, float[] argusEmbed, int[] inputShapes, string[] inputNames, string[] outputNames)
    {
        var ret = new List<PrismyV4Row>();
        var res = argusRunner.Run(new List<Array> { argusEmbed }, new List<int[]> { inputShapes }, inputNames.ToList(), outputNames.ToList());
        var argusLogits = res[0] as float[,];
        for (var i = 0; i < keys.Count; i++)
        {
            ret.Add(new PrismyV4Row(
                    keys[i],
                    TrimScore(argusLogits[i, 0]),
                    TrimScore(argusLogits[i, 1]),
                    TrimScore(argusLogits[i, 2]),
                    TrimScore(argusLogits[i, 3]),
                    TrimScore(argusLogits[i, 4]),
                    TrimScore(argusLogits[i, 5])
                    //TrimScore(argusLogits[i, 6]),
                    //TrimScore(argusLogits[i, 7])
                    ));
        }
        return ret;
    }
    public override IEnumerable<Row> Process(RowSet input, Row output, string[] args)
    {
        Init(args);
        List<string> keys = new List<string>();
        float[] argusEmbeddings = new float[batchSize * embedSize];

        var inputName = new string[1] { "input" };
        var outputName = new string[1] { "output" };
        var inputShapes = new int[2] { batchSize, embedSize };
        int idx = 0;
        foreach (Row row in input.Rows)
        {

            keys.Add(row["ImageKey"].String);
            var embed = row["AVEV9Vector_Binary_D1024"].Binary;
            Buffer.BlockCopy(embed, 0, argusEmbeddings, idx * embed.Length, embed.Length);
            idx += 1;
            if (idx == batchSize)
            {
                var ret = OutputRows(keys, argusEmbeddings, inputShapes, inputName, outputName);
                foreach (var o in ret)
                {
                    output.Reset();
                    output["ImageKey"].Set(o.ImageKey);
                    output["PrismyV4"].Set(o.PrismyV4);
                    output["EmotionalImpact"].Set(o.EmotionalImpact);
                    output["Inspiring"].Set(o.Inspiring);
                    output["Interestingness"].Set(o.Interestingness);
                    output["Aesthetic"].Set(o.Aesthetic);
                    output["PinStyle"].Set(o.PinStyle);
                    //output["Brightness"].Set(o.Brightness);
                    //output["Style"].Set(o.Style);
                    output["TimeStamp"].Set(Utils.GetTimestamp());
                    yield return output;
                }

                keys.Clear();
                idx = 0;
            }
        }

        if (idx > 0)
        {
            float[] restEmbeddings = new float[idx * embedSize];
            Array.Copy(argusEmbeddings, 0, restEmbeddings, 0, idx * embedSize);
            var ret = OutputRows(keys, restEmbeddings, new int[2] { idx, embedSize }, inputName, outputName);
            foreach (var o in ret)
            {
                output.Reset();
                output["ImageKey"].Set(o.ImageKey);
                output["PrismyV4"].Set(o.PrismyV4);
                output["EmotionalImpact"].Set(o.EmotionalImpact);
                output["Inspiring"].Set(o.Inspiring);
                output["Interestingness"].Set(o.Interestingness);
                output["Aesthetic"].Set(o.Aesthetic);
                output["PinStyle"].Set(o.PinStyle);
                //output["Brightness"].Set(o.Brightness);
                //output["Style"].Set(o.Style);
                output["TimeStamp"].Set(Utils.GetTimestamp());
                yield return output;
            }
        }
    }
}




public class KidFaceRow
{
    public string ImageKey { get; set; }
    public float KidScore { get; set; }
    public float FaceScore { get; set; }

    public KidFaceRow(string imageKey, float kidScore, float faceScore)
    {
        ImageKey = imageKey;
        KidScore = kidScore;
        FaceScore = faceScore;
    }
}

public class KidFaceProcessor : Processor
{
    int batchSize = 64;
    private OnnxWrapper.OnnxRunner argusRunner = null;
    private string embeddingPath = string.Empty;
    private int embedSize = 1024;
    private float max_val = 1.0f;
    private float min_val = 0.0f;

    private float TrimScore(float value)
    {
        return Math.Max(min_val, Math.Min(max_val, value));
    }

    public override bool RunSingleThreaded { get { return true; } }

    public override Schema Produces(string[] columns, string[] args, Schema input)
    {
        Schema schema = new Schema();
        schema.Add(new ColumnInfo("ImageKey", ColumnDataType.String));
        schema.Add(new ColumnInfo("KidScore", ColumnDataType.Float));
        schema.Add(new ColumnInfo("FaceScore", ColumnDataType.Float));
        return schema;
    }

    private void Init(string[] args)
    {
        if (argusRunner != null) return;
        var modelName = args[0];
        embedSize = int.Parse(args[1]);
        argusRunner = new OnnxWrapper.OnnxRunner(modelName);
    }

    private byte[] FloatArray2Bytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private float[] Bytes2FloatArray(byte[] data)
    {
        float[] values = new float[data.Length / sizeof(float)];
        Buffer.BlockCopy(data, 0, values, 0, data.Length);
        return values;
    }

    private List<KidFaceRow> OutputRows(List<string> keys, float[] argusEmbed, int[] inputShapes, string[] inputNames, string[] outputNames)
    {
        var ret = new List<KidFaceRow>();
        var res = argusRunner.Run(new List<Array> { argusEmbed }, new List<int[]> { inputShapes }, inputNames.ToList(), outputNames.ToList());
        var argusLogits = res[0] as float[,];
        for (var i = 0; i < keys.Count; i++)
        {
            ret.Add(new KidFaceRow(
                    keys[i],
                    TrimScore(argusLogits[i, 0]),
                    TrimScore(argusLogits[i, 1])
                    ));
        }
        return ret;
    }

    public override IEnumerable<Row> Process(RowSet input, Row output, string[] args)
    {
        Init(args);
        List<string> keys = new List<string>();
        float[] argusEmbeddings = new float[batchSize * embedSize];

        var inputName = new string[1] { "input" };
        var outputName = new string[1] { "output" };
        var inputShapes = new int[2] { batchSize, embedSize };
        int idx = 0;
        foreach (Row row in input.Rows)
        {

            keys.Add(row["ImageKey"].String);
            var embed = row["AVEV9Vector_Binary_D1024"].Binary;
            Buffer.BlockCopy(embed, 0, argusEmbeddings, idx * embed.Length, embed.Length);
            idx += 1;
            if (idx == batchSize)
            {
                var ret = OutputRows(keys, argusEmbeddings, inputShapes, inputName, outputName);
                foreach (var o in ret)
                {
                    output.Reset();
                    output["ImageKey"].Set(o.ImageKey);
                    output["KidScore"].Set(o.KidScore);
                    output["FaceScore"].Set(o.FaceScore);

                    yield return output;
                }

                keys.Clear();
                idx = 0;
            }
        }

        if (idx > 0)
        {
            float[] restEmbeddings = new float[idx * embedSize];
            Array.Copy(argusEmbeddings, 0, restEmbeddings, 0, idx * embedSize);
            var ret = OutputRows(keys, restEmbeddings, new int[2] { idx, embedSize }, inputName, outputName);
            foreach (var o in ret)
            {
                output.Reset();
                output["ImageKey"].Set(o.ImageKey);
                output["KidScore"].Set(o.KidScore);
                output["FaceScore"].Set(o.FaceScore);

                yield return output;
            }
        }
    }
}

