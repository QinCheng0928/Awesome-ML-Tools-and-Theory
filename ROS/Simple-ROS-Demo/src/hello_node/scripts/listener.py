# 1. Import libraries
import rospy
from hello_node.msg import Person

# Callback function to process received messages
def doPerson(p):
    rospy.loginfo("接收到的人的信息:%s, %d, %.2f",p.name, p.age, p.height)


if __name__ == "__main__":
    # 2. Initialize the ROS node with a unique name
    rospy.init_node("listener_person_p")
    # 3. Create a Subscriber object
    sub = rospy.Subscriber("chatter_person",Person,doPerson,queue_size=10)
    rospy.spin()