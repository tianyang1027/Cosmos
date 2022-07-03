using Microsoft.SCOPE.Types;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ScopeRuntime;
class MyClass
{
    public bool Isfilter(List<string> list, string imageUrl)
    {
        return list.Contains(imageUrl); 
    }
}
