"""
Simple test for the timeout module functionality.
This script validates that the basic timeout operations work correctly.
"""

import time
import logging
from faissx.client.timeout import timeout, set_timeout, TimeoutError

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# Test functions
@timeout(1.0)
def func_with_specific_timeout():
    """Function with a specific timeout value."""
    logger.info("Starting function with 1.0s timeout")
    time.sleep(0.5)  # Should complete before timeout
    logger.info("Completed successfully")
    return "Success"


@timeout
def func_with_global_timeout():
    """Function using the global timeout."""
    logger.info("Starting function with global timeout")
    time.sleep(0.2)  # Should complete before timeout
    logger.info("Completed successfully")
    return "Success"


class TimeoutExample:
    """Class with instance-level timeout."""

    def __init__(self):
        self.timeout = 2.0

    @timeout()
    def method_with_instance_timeout(self):
        """Method using the instance timeout attribute."""
        logger.info(f"Starting method with instance timeout {self.timeout}s")
        time.sleep(1.0)  # Should complete before timeout
        logger.info("Completed successfully")
        return "Success"


# Main test function
def run_tests():
    """Run a series of tests on the timeout module."""

    logger.info("=== Testing timeout module ===")

    # Test 1: Function with specific timeout
    try:
        result = func_with_specific_timeout()
        logger.info(f"Test 1 result: {result}")
    except TimeoutError as e:
        logger.error(f"Test 1 failed: {e}")

    # Test 2: Function with global timeout
    try:
        # Set global timeout to 1.5 seconds
        set_timeout(1.5)
        result = func_with_global_timeout()
        logger.info(f"Test 2 result: {result}")
    except TimeoutError as e:
        logger.error(f"Test 2 failed: {e}")

    # Test 3: Method with instance timeout
    try:
        example = TimeoutExample()
        result = example.method_with_instance_timeout()
        logger.info(f"Test 3 result: {result}")
    except TimeoutError as e:
        logger.error(f"Test 3 failed: {e}")

    logger.info("=== All tests completed ===")


if __name__ == "__main__":
    run_tests()
