import json
import os

#This helper class allows to access members of the configuration
#as c.attr1.attr2.attr3 instead of c[attr1][attr2][attr3]

class DictToAttr( object ):
    
    def __init__( self, dictionary ):
        
        self.dict = dict( dictionary )
    
    def __getattr__(self, attr):
        
        if attr in self.dict.keys():
            
            token = self.dict[ attr ]
            
            if isinstance( token, dict ):
                
                return DictToAttr( token )
            
            else:
                
                return token
        
        else:
            
            raise ValueError("'%s' does not exists in configuration" %(attr))
        
        

class Configuration( object ):
    
    def __init__(self):
        
        #Read configuration file
        
        path = os.path.dirname ( 
                  os.path.abspath( os.path.expanduser( __file__ ) )
                  )
        
        configFile = os.path.join( path, "configuration.json" )
        
        if os.path.exists( configFile ):
            
            try:
                
                with open( configFile ) as f:
            
                    #Strip all comments to make the output
                    #a json-compliant string
                    
                    allLines = f.readlines()
                    
                    clean = filter( lambda line: line.find("#") < 0, allLines )
                    
                    JSON = " ".join( clean )
                    JSON = JSON.replace("\n"," ")
                                
                self.config = json.loads( JSON )
            
            except IOError:
                
                print("Configuration file %s exists " % configFile + 
                      "but cannot be read. Traceback follows")
                
                raise
                
        else:
            
            raise IOError("Configuration file %s does not exist!" %(configFile))
    
    def __getattr__(self, attr):
        
        if attr in self.config.keys():
            
            token = self.config[ attr ]
            
            if isinstance( token, dict ):
                
                return DictToAttr( token )
            
            else:
                
                return token  
    
        
