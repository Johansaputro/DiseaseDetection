import unittest
from app import app

class AppTest(unittest.TestCase):
    def setUp(self):    
        # create a test client
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
    def tearDown(self):
        # remove the app context after testing
        self.app_context.pop()

    def test_is_alive(self):
        response = self.client.get('/isAlive')
        self.assertEqual(response.status_code, 200)

    def test_after_request(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')
        self.assertEqual(response.headers['Access-Control-Allow-Headers'], 'Content-Type,Authorization,Access-Control-Allow-Origin')
        self.assertEqual(response.headers['Access-Control-Allow-Methods'], 'GET,PUT,POST,DELETE,OPTIONS')
    
if __name__ == '__main__':
    unittest.main()