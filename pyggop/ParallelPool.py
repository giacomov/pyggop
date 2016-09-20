from pyggop.config import Configuration

try:

    from ipyparallel import Client

except ImportError:

    from IPython.parallel import Client


import multiprocessing

class ParallelPool( object ):
    
    def __init__(self):
        
        #Load configuration
        self.c = Configuration.Configuration(  )
        
        #Now instance the pool of batch workers according
        #to the technology selected in the configuration file
        
        if self.c.parallel.technology=='ipython':
            
            self.IPYc = Client( profile=self.c.parallel.ipython.profile )
    
            self.pool = self.IPYc[:]
        
        elif self.c.parallel.technology=='python':
            
            if self.c.parallel.python.number_of_processes==0:
                
                n_cpus = multiprocessing.cpu_count()
            
            else:
                
                n_cpus = self.c.parallel.python.number_of_processes
            
            self.pool = multiprocessing.Pool( n_cpus )
        
        else:
            
            raise ValueError("Unknown technology %s in configuration file" 
                             %(self.c.parallel.technology))
    
    #The following methods simply forward the requests to the
    #batch worker technology
    
    def map( self, *args, **kwargs ):
        
        if self.c.parallel.technology=='ipython':
        
            return self.pool.map( *args, **kwargs ).get()
        
        else:
            
            return self.pool.map( *args, **kwargs )
    
    def imap( self, *args, **kwargs ):
        
        return self.pool.imap( *args, **kwargs )
    
    def close( self ):
        
        if self.c.parallel.technology=='ipython':
            
            self.IPYc.close()
        
        else:
            
            self.pool.close()
            self.pool.join()
            
