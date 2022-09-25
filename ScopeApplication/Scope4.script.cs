using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using ScopeRuntime;
using Newtonsoft.Json.Linq;
using System.Linq;

// processor to further handle WrapStar output and return particular attributes
public class WrapStarProcessor : Processor
{
    public override Schema Produces(string[] columns, string[] args, Schema input)
    {
        return new Schema("PageUrl,ImageUrl,Title,Width,Heigth,Description");
    }

    private static int GetColumnIndex(Schema schema, string name)
    {
        return schema.Contains(name) ? schema.IndexOf(name) : -1;
    }

    public override IEnumerable<Row> Process(RowSet input, Row output, string[] args)
    {
        int idxModelJson = GetColumnIndex(input.Schema, "Model_Json");
        foreach (Row row in input.Rows)
        {
            if (idxModelJson >= 0)
            {
                string model_jsonStr = row["Model_Json"].ToString();
                string pageUrl = row["DocumentURL"].ToString();
                if (JObject.Parse(model_jsonStr)["Entities"].Count() > 0 && JObject.Parse(model_jsonStr)["Entities"][0]["Properties"].Count() > 0)
                {
                    var description = JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][0]["Value"][0].ToString();
                    var title = JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][1]["Value"][0].ToString();
                    var url = JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][2]["Value"][0].ToString();
                    var height = JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][3]["Value"][0].ToString();
                    var width = JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][4]["Value"][0].ToString();
                    var community = title.Split('-')[0].Replace("r/");
                    output[0].Set(pageUrl);
                    output[1].Set(url);
                    output[2].Set(title);
                    output[3].Set(width);
                    output[4].Set(height);
                    output[5].Set(description);
                    output[6].Set(description);
                    output[7].Set(description);
                    output[8].Set(description);
                }
                yield return output;

            }
            else
            {
                throw new Exception("No WrapStar data available");
            }
        }
    }
}

