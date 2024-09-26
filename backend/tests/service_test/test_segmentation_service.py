import unittest
from app import app
from services.segmentation_service import SegmentationService

class SegmentationServiceTest(unittest.TestCase):
    def setUp(self):    
        self.segmentation_service = SegmentationService()
        
    def tearDown(self):
        # remove the app context after testing
        self.app_context.pop()