import unittest
from vsa.hrr import HRR

class TestEncoderMethods(unittest.TestCase):

    def test_encoder(self):
    
        # test chars
    	a, b = HRR('a'), HRR('b')
    	self.assertItemsEqual(a.memory, HRR.mapping['a'])
    	self.assertItemsEqual(b.memory, HRR.mapping['b'])
        for i in range(len(a.memory)):
            self.assertNotEqual(a.memory[i], HRR.mapping['b'][i])
        
        # test scalar
        a, b = HRR(5), HRR(7)
    	self.assertItemsEqual(a.memory, HRR(5).memory)
    	self.assertItemsEqual(b.memory, HRR(7).memory)
        for i in range(len(a.memory)):
            self.assertNotEqual(a.memory[i], HRR(7).memory[i])
        
        d = HRR(3+2)
        self.assertItemsEqual(a.memory, d.memory) # parameter in constructor should be treated as single int
        
    def test_binding(self):
        
        av, bv = 5, 50;
        a, b = HRR(av), HRR(bv)
        
        # bind and probe
        t1 = a * b
        t1 = t1 / b
        self.assertTrue((av-1) in t1 or av in t1 or (av+1) in t1)
        
        t2 = (a * b) / b
        self.assertTrue((av-1) in t2 or av in t2 or (av+1) in t2)
        self.assertFalse((bv-1) in t2 or bv in t2 or (bv+1) in t2)
        
        t3 = (a * bv) / b
        self.assertTrue((av-1) in t2 or av in t2 or (av+1) in t2)
        
    def test_decoder(self):
        
        av, bv = 3, 33;
        a, b = HRR(av), HRR(bv)
        
        t1 = (a * b) % b
        m_before = t1.memory
        t2 = t1.decode()
        m_after = t1.memory
        self.assertTrue((av-1) in t2 or av in t2 or (av+1) in t2) # decoding proper value?
        self.assertItemsEqual(m_before, m_after) # not changing inner representation
        
if __name__ == '__main__':
    unittest.main()