using Microsoft.SCOPE.Types;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ScopeRuntime;
class MyClass
{
    public static string GetLatestDate(string path)
    {
        if(string.IsNullOrEmpty(path)) return "";
        var strs=path.Split('/');
        if(strs.Length>8) return strs[8];
        return "";
    }

}