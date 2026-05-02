from setuptools import find_packages, setup

package_name = 'electricity_monitor'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Matchabo Adrien',
    maintainer_email='matchaboaubert74@email.com',
    description='Simulation capteurs pannes électricité — Hôpital camerounais',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sensor_publisher = electricity_monitor.sensor_publisher:main',
            'data_collector   = electricity_monitor.data_collector:main',
        ],
    },
)
