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
        return new Schema("PageUrl,ImageUrl,Title,Width,Heigth,Description,Community,Vote,Comment");
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
                var propertiesCount = JObject.Parse(model_jsonStr)["Entities"].Count() > 0 ? JObject.Parse(model_jsonStr)["Entities"][0]["Properties"].Count() : 0;
                if (JObject.Parse(model_jsonStr)["Entities"].Count() > 0 && propertiesCount > 0)
                {
                    var description = propertiesCount > 0 ? JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][0]["Value"][0].ToString() : "";
                    var title = propertiesCount > 1 ? JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][1]["Value"][0].ToString() : "";
                    var url = propertiesCount > 2 ? JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][2]["Value"][0].ToString() : "";
                    var height = propertiesCount > 3 ? JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][3]["Value"][0].ToString() : "";
                    var width = propertiesCount > 4 ? JObject.Parse(model_jsonStr)["Entities"][0]["Properties"][4]["Value"][0].ToString() : "";
                    var vote = !string.IsNullOrEmpty(description) ? description.Contains("votes") ? description.Split(' ')[0] : "0" : "0";
                    var comment = !string.IsNullOrEmpty(description) ? description.Contains("comment") ? description.Split(' ')[3] : "0" : "0";
                    var community = !string.IsNullOrEmpty(title) ? title.Split('-').Length > 0 ? title.Split('-')[0].Replace("r/", "") : "" : "";
                    output[0].Set(pageUrl);
                    output[1].Set(url);
                    output[2].Set(title);
                    output[3].Set(width);
                    output[4].Set(height);
                    output[5].Set(description);
                    output[6].Set(community);
                    output[7].Set(vote);
                    output[8].Set(comment);
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
