/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	double sample_x, sample_y, sample_theta;
	num_particles = 100;
	default_random_engine gen;
	Particle temp_particle;

	// Create the normal distributions
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	sample_x = dist_x(gen);
	sample_y = dist_y(gen);
	sample_theta = dist_theta(gen);
	for (int ii = 0; ii < num_particles; ++ii){
		temp_particle.id = ii;
		temp_particle.x = sample_x;
		temp_particle.y = sample_y;
		temp_particle.theta = sample_theta;
		temp_particle.weight = 1.0;
		particles.push_back(temp_particle);
		//cout<<particles[ii].id<<endl;
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
		
	for (int ii = 0; ii < num_particles; ++ii){
		double new_x, new_y, new_theta;
		
		//take care of zero yaw rate
		if (yaw_rate < 0.00001 && yaw_rate > -0.00001){
			new_x = particles[ii].x + velocity * cos(particles[ii].theta) * delta_t;
			new_y = particles[ii].y + velocity * sin(particles[ii].theta) * delta_t;
		}
		else{//follow the notes
			new_x = particles[ii].x + velocity / yaw_rate * (sin(particles[ii].theta + yaw_rate * delta_t) - sin(particles[ii].theta));
			new_y = particles[ii].y + velocity / yaw_rate * (cos(particles[ii].theta) - cos(particles[ii].theta + yaw_rate * delta_t));
		
		}
		new_theta = particles[ii].theta + yaw_rate * delta_t;
	
		// Create the normal distributions
		normal_distribution<double> dist_x(new_x, std_pos[0]);
		normal_distribution<double> dist_y(new_y, std_pos[1]);
		normal_distribution<double> dist_theta(new_theta, std_pos[2]);

		// Sample from the normal distribution
		particles[ii].x = dist_x(gen);
		particles[ii].y = dist_y(gen);
		particles[ii].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double dist_min = 1000000;
	//LandmarkObs associated;
	//vector<LandmarkObs> associated_list;
	for (unsigned int i=0; i < observations.size(); ++i){ // for each observation 
		for(unsigned int j = 0; j < predicted.size(); ++j){ // find the associated label of the map
			double distance;
			distance =dist(observations[j].x,observations[j].y, predicted[i].x,predicted[i].y);  
			//cout<<"dist: "<<dist<<endl;
			if (distance < dist_min){
				dist_min = distance;
				observations[i].id	 = predicted[i].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	double total_weight = 0;
	vector<double> particle_weights;
	
	//run through each particle
	for (unsigned int ii = 0; ii < particles.size(); ++ii){
		LandmarkObs obs_map;
		LandmarkObs pred;
		vector<LandmarkObs> obs_map_coords;
		//vector for holding the map data around a particle
		vector<LandmarkObs> predicted;
		
		//create a vector of landmarks in map coordinates. particles are in map coordinates
		for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); ++i){
			pred.id = map_landmarks.landmark_list[i].id_i;
			pred.x = map_landmarks.landmark_list[i].x_f;
			pred.y = map_landmarks.landmark_list[i].y_f;

			//calculate the distance from the particle to the map landmarks
			double distance = dist(pred.x,pred.y,particles[ii].x,particles[ii].y);
			//add the Map Landmark to the vector if within sensor range
			if (distance < sensor_range){
				predicted.push_back(pred);
			}
		}	

		//transform car observations to map coordiates
		for (unsigned int jj = 0; jj < observations.size(); ++jj){
			obs_map.id = observations[jj].id;
			obs_map.x = particles[ii].x + observations[jj].x * cos(particles[ii].theta) - sin(particles[ii].theta) * observations[jj].y;
			obs_map.y = particles[ii].y + observations[jj].x * sin(particles[ii].theta) + cos(particles[ii].theta) * observations[jj].y;
			obs_map_coords.push_back(obs_map);
		}

		//calculate the weight for each observation
		//set initial weight for the particle to 1
		double temp_weight = 1;
		//for each observation
		for (unsigned int k = 0; k < obs_map_coords.size(); ++k){
			double x_obs_map = obs_map_coords[k].x;
			double y_obs_map = obs_map_coords[k].y;
			double dist_min = 10000000;
			double weight_min = 1;
			//check the landmark distance to the observation
			for(unsigned int j = 0; j < predicted.size(); ++j){ // find the associated label of the map
				double distance;
				double mu_x;
				double mu_y;
				distance =dist(x_obs_map,y_obs_map, predicted[j].x,predicted[j].y);  
				
				//if the distance between the landmark and the observation smaller than the last landmark
				if (distance < dist_min){
					//assign the landmark id to the observation
					obs_map_coords[k].id = predicted[j].id;
					//assign the landmark location for calculating the multivariate gaussian probability
					mu_x = predicted[j].x;
					mu_y = predicted[j].y;
					//make the current distance the min
					dist_min = distance;
					//calculate the weight
					weight_min = 1/2.0/M_PI/std_landmark[0]/std_landmark[1]*exp(-((x_obs_map-mu_x)*(x_obs_map-mu_x)/2.0/std_landmark[0]/std_landmark[0]+(y_obs_map-mu_y)*(y_obs_map-mu_y)/2.0/std_landmark[1]/std_landmark[1]));  
				}
			}//finish cycling thru the landmarks (we should have the smallest distance weight now)
			//multiply the current particle weight by the weight of the kth observation made
			temp_weight = temp_weight * weight_min;
			
		}//finish checking all the observations
		//assign the product of weights to the current particle weight
		particle_weights.push_back(temp_weight);
		//sum the weights of all the particles.  will use this to normalize the weights
		total_weight+=temp_weight;
	}
	//normalize the weights and update the particle weights
	for(unsigned int i = 0; i < particle_weights.size(); ++i){
		particles[i].weight = particle_weights[i] / total_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//recreate the python code from Sebastian's lecture
	uniform_int_distribution<int> ind_particles (1,num_particles);
	default_random_engine gen;
	unsigned int index = ind_particles(gen);//some random function here
	double random_one = ((double) rand() / (RAND_MAX));//this can improved
	double beta = 0.0;
	float mw = 0;
	
	//get the largest weight
	for(unsigned int i = 0; i < particles.size(); ++i){
		if (particles[i].weight > mw){
			mw = particles[i].weight;
		}
	}

	//make a new vector of particles from the old particles
	vector<Particle> new_particles = particles;
	
	//do the resample
	for(int i = 0; i < num_particles; ++i){
		beta+= 2.0 * mw * random_one;// some random number
		while(beta > particles[index].weight){
			beta-=particles[index].weight;
			index = (index + 1) % num_particles;
		}
		new_particles[i] = particles[index];
	}

	//re-assign the new_particles to particles so you have the resampled list of particles
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
