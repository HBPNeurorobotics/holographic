import unittest
from hrr import HRR

class TestEncoderMethods(unittest.TestCase):

    def test_encoder(self):
    
        # test chars
    	a, b = HRR('a'), HRR('b')
    	self.assertEqual(a.memory, HRR.mappings['a'])
    	self.assertEqual(b.memory, HRR.mappings['b'])
    	self.assertNotEqual(a.memory, HRR.mappings['b'])
        
        # test scalar
        a, b = HRR(5), HRR(7)
    	self.assertEqual(a.memory, HRR.mappings[5])
    	self.assertEqual(b.memory, HRR.mappings[7])
    	self.assertNotEqual(b.memory, HRR.mappings[5])
        
        d = HRR(3+2)
        self.assertEqual(a, d) # parameter in constructor should be treated as single int
        
    def test_binding(self):
        
        a, b = HRR(5), HRR(10)
        
        # bind and probe
        t1 = a * b
        t1 = t1 / b
        self.assertEqual(t1, a)
        
        t2 = (a * b) / b
        self.assertEqual(t2, a)
        self.assertNotEqual(t2, b)
        
        t3 = (a * 10) / b
        self.assertEqual(t3, a)
        
    def test_decoder(self):
        
        a, b = HRR(3), HRR(33)
        
        t1 = (a * b) % b
        m_before = t1.memory
        t1 = t1.decode()
        m_after = t1.memory
        self.assertEqual(t1, 3) # decoding proper value?
        self.assertEqual(m_before, m_after) # not changing inner representation
        
if __name__ == '__main__':
    unittest.main()