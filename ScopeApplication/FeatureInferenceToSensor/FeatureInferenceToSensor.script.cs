using Microsoft.SCOPE.Types;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ScopeRuntime;
using System.Linq;

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