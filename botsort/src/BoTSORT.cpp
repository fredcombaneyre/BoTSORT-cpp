#include "BoTSORT.h"

#include <optional>
#include <unordered_set>
#include <iostream>

// #include <opencv2/imgproc.hpp>

#include "DataType.h"
// #include "INIReader.h"
#include "matching.h"
// #include "profiler.h"
#include <map>

namespace bot_sort
{

// LOGGING MACROS FOR TRACING
//
#define LOG_START if (_trace){ std::cout << " ----------------------- BoTSORT trace -------------------------" << std::endl;}
#define LOG_END   if (_trace) {std::cout << " -------------------- END of BoTSORT trace ---------------------" << std::endl;}
#define LOG_ITEM(item) if (_trace) { std::cout << item ; }
#define LOG_INIT_COUNT if (_trace) { _ic=0; }
#define LOG_ITEM_NEW_LINE(count) if (_trace && _ic>=count) {LOG_INIT_COUNT;std::cout << std::endl;}
#define LOG_ADD_COMMA if (_trace && _ic == 0) { std::cout << "  ==> trackIDs= ";} else { std::cout << ",";} _ic++
#define LOG_ITEM_END_LINE if (_trace && _ic>0) {std::cout << std::endl;}  

#define LOG_TITLE_STEP(frame_id, step, title) if (_trace) {std::cout << "  F=" << frame_id \
                        << ": ==================== Step " << step << ": " << title << " ====================" << std::endl;}  
#define LOG_TITLE_ARG(title, ...) if (_trace) { char buf[1024]; sprintf(buf, title, ##__VA_ARGS__); \
                                                     std::cout << "---------- " << std::string(buf) << " ----------" << std::endl;}

#define LOG_TRACK_DET_DISTANCE(tracks, detections, dist_matrix) \
                        if (_trace) { \
                            for(uint i=0 ; i<dist_matrix.rows() ; i++){ \
                                std::cout << "[" << tracks[i]->getTrackId() << "] => " ; \
                                for(uint j=0;j<dist_matrix.cols();j++) { \
                                    if (dist_matrix(i,j) < 1) { \
                                        std::cout << " (" << detections[j]->getDetId() << ")=" << dist_matrix(i,j) << "," ; \
                                    } \
                                } \
                                std::cout << std::endl; \
                            } \
                        }
#define LOG_ASSIGNMENT(track, detection,type) if (_trace) {LOG_ADD_COMMA; \
                                    std::cout << " " << type \
                                    << "[" << track->getTrackId() << "]=(" << detection->getDetId() << ")" ;}  
#define LOG_TRACK(track) if (_trace) { LOG_ADD_COMMA; std::cout << "[" << track->getTrackId() << "]"; }
#define LOG_TRACK_T(track,type) if (_trace) { LOG_ADD_COMMA; std::cout << " " << type << "[" << track->getTrackId() << "]"; }
#define LOG_DETECTION(detection,type) if (_trace) {std::cout << "   " << type << "," \
                        << " detID=" << tracklet->getDetId() << " [ " \
                        << tracklet->get_tlwh()[0] << " , " << tracklet->get_tlwh()[1] \
                        << " , " << tracklet->get_tlwh()[2] << " , " << tracklet->get_tlwh()[3] << " ]" \
                        << " score=" << tracklet->get_score() << std::endl;\
                    }
#define LOG_TRACKS_DETAILS(tracks) if (_trace) { \
                                    for (auto &track : tracks){ \
                                        std::cout << "  ==> trackID=" << track->getTrackId() << " [ " \
                                        << track->get_tlwh()[0] << " , " << track->get_tlwh()[1] \
                                        << " , " << track->get_tlwh()[2] << " , " << track->get_tlwh()[3] \
                                        << " ] score=" << track->get_score() << std::endl; \
                                    } \
                                }                    
#define LOG_DUPLICATE(trackD, trackO) if (trace) { \
                                    std::cout << "Removing duplicate track [" << trackD->getTrackId() << "]" \
                                    << " keeping [" << trackO->getTrackId() << "]" << std::endl;}
//
// END - LOGGING MACROS FOR TRACING

template<typename T>
bool requires_load(const Config<T> &config)
{
    return std::holds_alternative<std::string>(config) &&
           !std::get<std::string>(config).empty();
}

template<typename T>
bool not_empty(const Config<T> &config)
{
    bool has_config = !std::holds_alternative<std::monostate>(config);
    bool is_non_empty = !(std::holds_alternative<std::string>(config) &&
                          std::get<std::string>(config).empty());

    return has_config && is_non_empty;
}

template<typename T>
T fetch_config(const Config<T> &config,
               std::function<T(const std::string &)> loader)
{
    if (std::holds_alternative<T>(config))
    {
        return std::get<T>(config);
    }

    if (requires_load(config))
    {
        return loader(std::get<std::string>(config));
    }

    throw std::runtime_error("Config is empty");
}

template<typename T>
T get_config(const Config<T> &config)
{
    if (std::holds_alternative<T>(config))
    {
        return std::get<T>(config);
    }

    throw std::runtime_error("Config is empty");
}

BoTSORT::BoTSORT(const Config<TrackerParams> &tracker_config,
                            const Config<GMC_Params> &gmc_config,
                            const Config<ReIDParams> &reid_config,
                            const std::string &reid_onnx_model_path,
                            bool trace) : _trace(trace)
{
    // auto tracker_params = fetch_config<TrackerParams>(tracker_config, TrackerParams::load_config);
    auto tracker_params = get_config<TrackerParams>(tracker_config);
    _load_params_from_config(tracker_params);

    // Tracker module
    _frame_id = 0;
    // _buffer_size = static_cast<uint8_t>(_frame_rate / 30.0 * _track_buffer);
    _buffer_size = static_cast<uint8_t>(_track_buffer);
    _max_time_lost = _buffer_size;
    _kalman_filter = std::make_unique<KalmanFilter>(static_cast<double>(1.0 / _frame_rate));

    // Re-ID module, load visual feature extractor here
    if (_reid_enabled && not_empty(reid_config) && reid_onnx_model_path.size() > 0){

        // auto reid_params = fetch_config<ReIDParams>(reid_config, ReIDParams::load_config);
        // _reid_model = std::make_unique<ReIDModel>(reid_params, reid_onnx_model_path); 
        auto reid_params = get_config<ReIDParams>(reid_config);
        _reid_distance_metric = reid_params.distance_metric;
    } else {
        // std::cout << "Re-ID module disabled" << std::endl;
        _reid_enabled = false;
    }

    // Global motion compensation module
    if (_gmc_enabled && not_empty(gmc_config)) {
        // auto gmc_params = fetch_config<GMC_Params>(gmc_config, [this](const std::string &config_path) {
        //     return GMC_Params::load_config(GlobalMotionCompensation::GMC_method_map[_gmc_method_name],config_path);});
        // _gmc_algo = std::make_unique<GlobalMotionCompensation>(gmc_params);
    }
    else {
        // std::cout << "GMC disabled" << std::endl;
        _gmc_enabled = false;
    }
}

std::vector<std::shared_ptr<Track>>
BoTSORT::track(const std::vector<Detection> &detections) {
// BoTSORT::track(const std::vector<Detection> &detections, const cv::Mat &frame)

    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////
    // For all detections, extract features, create tracks and classify on the segregate of confidence
    _frame_id++;
    int _ic=0; // used for logging purposes to count the number of items printed in a line for better formatting

    LOG_START
    LOG_TITLE_STEP(_frame_id, "0", "Get detections");
    LOG_TITLE_ARG("L < score=%.02f <= H", _track_high_thresh)

    std::vector<std::shared_ptr<Track>> activated_tracks, refind_tracks;
    std::vector<std::shared_ptr<Track>> detections_high_conf, detections_low_conf;
    detections_low_conf.reserve(detections.size()),
            detections_high_conf.reserve(detections.size());

    if (!detections.empty())
    {
        for (Detection &detection:
             const_cast<std::vector<Detection> &>(detections))
        {
            detection.bbox_tlwh.x = std::max(0.0f, detection.bbox_tlwh.x);
            detection.bbox_tlwh.y = std::max(0.0f, detection.bbox_tlwh.y);
            // detection.bbox_tlwh.width = std::min(static_cast<float>(frame.cols - 1),detection.bbox_tlwh.width);
            // detection.bbox_tlwh.height =std::min(static_cast<float>(frame.rows - 1),detection.bbox_tlwh.height);
            detection.bbox_tlwh.width = std::max(0.0f, detection.bbox_tlwh.width);
            detection.bbox_tlwh.height = std::max(0.0f, detection.bbox_tlwh.height);

            std::shared_ptr<Track> tracklet;
            std::vector<float> tlwh = {
                    detection.bbox_tlwh.x, detection.bbox_tlwh.y,
                    detection.bbox_tlwh.width, detection.bbox_tlwh.height};

            if (detection.confidence > _track_low_thresh)
            {
                if (_reid_enabled) {
                    // FeatureVector embedding = _extract_features(frame, detection.bbox_tlwh);
                    FeatureVector embedding = _get_features(detection.featArray, detection.featDim);
                    tracklet = std::make_shared<Track>(tlwh, detection.confidence, detection.class_id, detection.detId, embedding);
                } else {
                    tracklet = std::make_shared<Track>( tlwh, detection.confidence, detection.class_id, detection.detId);
                }

                if (detection.confidence >= _track_high_thresh) {
                    detections_high_conf.push_back(tracklet);
                    LOG_DETECTION(tracklet,"H")
                } else {
                    detections_low_conf.push_back(tracklet);
                    LOG_DETECTION(tracklet,"L")
                }
            }
        }
    }

    // Segregate tracks in unconfirmed and tracked tracks
    std::vector<std::shared_ptr<Track>> unconfirmed_tracks, tracked_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks)
    {
        if (!track->is_activated)
        {
            unconfirmed_tracks.push_back(track);
        }
        else
        {
            tracked_tracks.push_back(track);
        }
    }
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////


    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////
    // Merge currently tracked tracks and lost tracks
    std::vector<std::shared_ptr<Track>> tracks_pool;
    tracks_pool = _merge_track_lists(tracked_tracks, _lost_tracks);

    // Predict the location of the tracks with KF (even for lost tracks)
    Track::multi_predict(tracks_pool, *_kalman_filter);

    // Estimate camera motion and apply camera motion compensation
    if (_gmc_enabled)
    {
        // HomographyMatrix H = _gmc_algo->apply(frame, detections);
        // Track::multi_gmc(tracks_pool, H);
        // Track::multi_gmc(unconfirmed_tracks, H);
    }
    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////
    
    LOG_TITLE_ARG("Active tracks")
    LOG_TRACKS_DETAILS(tracked_tracks)
    LOG_TITLE_ARG("Lost tracks")
    LOG_TRACKS_DETAILS(_lost_tracks)
    
    ////////////////// ASSOCIATION ALGORITHM STARTS HERE //////////////////
    ////////////////// First association, with high score detection boxes //////////////////
    // Find IoU distance between all tracked tracks and high confidence detections
    CostMatrix iou_dists, raw_emd_dist, iou_dists_mask_1st_association, emd_dist_mask_1st_association;
    CostMatrix distances_first_association;

    std::tie(iou_dists, iou_dists_mask_1st_association) = iou_distance(tracks_pool, detections_high_conf, _proximity_thresh);
    fuse_score(iou_dists,detections_high_conf);// Fuse the score with IoU distance

    LOG_TITLE_STEP(_frame_id, "1", "First association, High score tracks with IoU+Score");
    LOG_TITLE_ARG("IoU distances < 1 -- [track]=> (det)=dist fuse(iou+score) ")
    
    LOG_TRACK_DET_DISTANCE(tracks_pool, detections_high_conf, iou_dists)
    
    if (_reid_enabled)
    {
        // If re-ID is enabled, find the embedding distance between all tracked tracks and high confidence detections
        std::tie(raw_emd_dist, emd_dist_mask_1st_association) =
                    embedding_distance(tracks_pool, detections_high_conf,_appearance_thresh,_reid_distance_metric, 2);
                // embedding_distance(tracks_pool, detections_high_conf,_appearance_thresh,_reid_model->get_distance_metric());
                

        LOG_TITLE_ARG("Embedding distances -- [track]=> (det)=dist ( _appearance_thresh=%.02f )", _appearance_thresh)
        LOG_TRACK_DET_DISTANCE(tracks_pool, detections_high_conf, raw_emd_dist)

        // Fuse the IoU distance and embedding distance to get the final distance matrix
        // IoU and emb masks will be applied on emb_dists, and we will take the min(iou_dists, emb_dists)
        distances_first_association = fuse_iou_with_emb(iou_dists, raw_emd_dist, 
                                        iou_dists_mask_1st_association, emd_dist_mask_1st_association);
        
        // // Popular ReID method (JDE / FairMOT)        
        // std::tie(raw_emd_dist, emd_dist_mask_1st_association) =
        //         // embedding_distance(tracks_pool, detections_high_conf,_appearance_thresh,_reid_model->get_distance_metric());
        //         embedding_distance(tracks_pool, detections_high_conf,_appearance_thresh,_reid_distance_metric, 1); 
        // fuse_motion(*_kalman_filter, raw_emd_dist, tracks_pool, detections_high_conf, _lambda);// Fuse the motion with embedding distance 
        // distances_first_association = raw_emd_dist;  
        
        // // IoU masking ReID
        // std::tie(raw_emd_dist, emd_dist_mask_1st_association) =
        //         // embedding_distance(tracks_pool, detections_high_conf,_appearance_thresh,_reid_model->get_distance_metric());
        //         embedding_distance(tracks_pool, detections_high_conf,_appearance_thresh,_reid_distance_metric, 1); 
        // CostMatrix dummy_empty_mat; // using dummy empty matrix to just apply iou_dists_mask_1st_association over raw_emd_dist
        // distances_first_association = fuse_iou_with_emb(
        //         raw_emd_dist, dummy_empty_mat, iou_dists_mask_1st_association, dummy_empty_mat);

    } else { // IOU only
        // Fuse the IoU distance and embedding distance to get the final distance matrix
        // We are not using ReID here so raw_emd_dist is empty and we are just appliying iou mask over iou_dists. 
        distances_first_association = fuse_iou_with_emb(iou_dists, raw_emd_dist, 
                                    iou_dists_mask_1st_association, emd_dist_mask_1st_association);
    }

    LOG_TITLE_ARG("IoU distances < 1 [track]=> (det)=dist fuse(iou+emb) ")
    LOG_TRACK_DET_DISTANCE(tracks_pool, detections_high_conf, distances_first_association);
    

    // Perform linear assignment on the final distance matrix, LAPJV algorithm is used here
    AssociationData first_associations = linear_assignment(distances_first_association, _match_thresh);

    LOG_TITLE_ARG("IoU assignments with High detections (IoU < match_thresh=%.02f)", _match_thresh)
    LOG_INIT_COUNT

    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: first_associations.matches)
    {
        const std::shared_ptr<Track> &track = tracks_pool[match.first];
        const std::shared_ptr<Track> &detection = detections_high_conf[match.second];

        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked)
        {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
            LOG_ASSIGNMENT(track, detection, "T")
        }
        else
        {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
            LOG_ASSIGNMENT(track, detection, "R")
        }
        LOG_ITEM_NEW_LINE(8)
    }
    
    LOG_ITEM_END_LINE;
    
    ////////////////// First association, with high score detection boxes //////////////////


    ////////////////// Second association, with low score detection boxes //////////////////
    LOG_TITLE_ARG("Unmatched tracks after 1st association with high score ('L' for lost tracks)")
    LOG_INIT_COUNT
    // Get all unmatched but tracked tracks after the first association, these tracks will be used for the second association
    std::vector<std::shared_ptr<Track>> unmatched_tracks_after_1st_association;
    for (int track_idx: first_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track = tracks_pool[track_idx];
        if (track->state == TrackState::Tracked)
        {
            unmatched_tracks_after_1st_association.push_back(track);
            LOG_TRACK(track);
        } else {
            LOG_TRACK_T(track,"L") // L for lost track that is unmatched after the first association;
        }
        LOG_ITEM_NEW_LINE(8)
    }
    LOG_ITEM_END_LINE;


    LOG_TITLE_STEP(_frame_id, "2", "Second association, using low score dets");
    

    // Find IoU distance between unmatched but tracked tracks left after the first association and low confidence detections
    CostMatrix iou_dists_second;
    iou_dists_second = iou_distance(unmatched_tracks_after_1st_association,detections_low_conf);

    LOG_TITLE_ARG("IoU distances < 1 -- [track]=> (det)=dist ")
    LOG_TRACK_DET_DISTANCE(unmatched_tracks_after_1st_association, detections_low_conf, iou_dists_second);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData second_associations = linear_assignment(iou_dists_second, 0.5);

    LOG_TITLE_ARG("IoU assignments with Low detections - match_thresh=0.5 (fixed) ")
    LOG_INIT_COUNT

    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: second_associations.matches)
    {
        const std::shared_ptr<Track> &track = unmatched_tracks_after_1st_association[match.first];
        const std::shared_ptr<Track> &detection = detections_low_conf[match.second];

        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked) {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
            LOG_ASSIGNMENT(track, detection, "T")

        } else {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
            LOG_ASSIGNMENT(track, detection, "R")
        }
    }

    LOG_ITEM_END_LINE;


    LOG_TITLE_ARG("Unmatched new lost tracks")
    LOG_INIT_COUNT
    // The tracks that are not associated with any detection even after the second association are marked as lost
    std::vector<std::shared_ptr<Track>> lost_tracks;
    for (int unmatched_track_index: second_associations.unmatched_track_indices) {
        const std::shared_ptr<Track> &track = unmatched_tracks_after_1st_association[unmatched_track_index];

        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_tracks.push_back(track);
            LOG_TRACK(track);
        }
    }
    LOG_ITEM_END_LINE
    ////////////////// Second association, with low score detection boxes //////////////////

    LOG_TITLE_STEP(_frame_id, "3", "Third association, unconfirmed tracks with high score detections");

    ////////////////// Deal with unconfirmed tracks //////////////////
    std::vector<std::shared_ptr<Track>>
            unmatched_detections_after_1st_association;
    for (int detection_idx: first_associations.unmatched_det_indices) {
        const std::shared_ptr<Track> &detection = detections_high_conf[detection_idx];
        unmatched_detections_after_1st_association.push_back(detection);
    }

    //Find IoU distance between unconfirmed tracks and high confidence detections left after the first association
    CostMatrix iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
            iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed;

    std::tie(iou_dists_unconfirmed, iou_dists_mask_unconfirmed) =
                iou_distance( unconfirmed_tracks, unmatched_detections_after_1st_association, _proximity_thresh);
    fuse_score(iou_dists_unconfirmed, unmatched_detections_after_1st_association);

    LOG_TITLE_ARG("IoU distances < 1 ( [det]=fuse(iou+score) )")
    LOG_TRACK_DET_DISTANCE(unconfirmed_tracks, unmatched_detections_after_1st_association, iou_dists_unconfirmed)
     

    if (_reid_enabled)
    {
        // Find embedding distance between unconfirmed tracks and high confidence detections left after the first association
        std::tie(raw_emd_dist_unconfirmed, emd_dist_mask_unconfirmed) =
                // embedding_distance(unconfirmed_tracks,unmatched_detections_after_1st_association,_appearance_thresh,_reid_model->get_distance_metric());
                embedding_distance(unconfirmed_tracks, unmatched_detections_after_1st_association, _appearance_thresh,_reid_distance_metric, 2);

        // fuse_motion(*_kalman_filter, raw_emd_dist_unconfirmed, unconfirmed_tracks, unmatched_detections_after_1st_association, _lambda);
    }

    // Fuse the IoU distance and the embedding distance
    CostMatrix distances_unconfirmed = fuse_iou_with_emb(
            iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
            iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed);

    LOG_TITLE_ARG("IoU distances < 1 -- [track]=> (det)=dist fuse(iou+score+emb)  --------- ")
    LOG_TRACK_DET_DISTANCE(unconfirmed_tracks, unmatched_detections_after_1st_association, distances_unconfirmed)
              

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData unconfirmed_associations = linear_assignment(distances_unconfirmed, 0.7);

    LOG_TITLE_ARG("IoU assignments with High detections - match_thresh=0.7 (fixed) ")
    LOG_INIT_COUNT
    
    for (const std::pair<int, int> &match: unconfirmed_associations.matches)
    {
        const std::shared_ptr<Track> &track = unconfirmed_tracks[match.first];
        const std::shared_ptr<Track> &detection =
                unmatched_detections_after_1st_association[match.second];

        // If the unconfirmed track is associated with a detection we update the track with the new associated detection
        // and add the track to the activated tracks list
        track->update(*_kalman_filter, *detection, _frame_id);
        activated_tracks.push_back(track);
        LOG_ASSIGNMENT(track, detection, "A")
        LOG_ITEM_NEW_LINE(8)
    }

    LOG_ITEM_END_LINE

    LOG_TITLE_ARG("Removing unconfirmed tracks")
    LOG_INIT_COUNT


    // All the unconfirmed tracks that are not associated with any detection are marked as removed
    std::vector<std::shared_ptr<Track>> removed_tracks;
    for (int unmatched_track_index:unconfirmed_associations.unmatched_track_indices){
        const std::shared_ptr<Track> &track = unconfirmed_tracks[unmatched_track_index];
        track->mark_removed();
        removed_tracks.push_back(track);

        LOG_TRACK(track);
    }

    LOG_ITEM_END_LINE
    ////////////////// Deal with unconfirmed tracks //////////////////


    ////////////////// Initialize new tracks //////////////////
    LOG_TITLE_STEP(_frame_id, "4", "Initialize new tracks ");    
    LOG_TITLE_ARG("Init only tracks with score >= new_track_thresh=%.02f", _new_track_thresh)
    LOG_INIT_COUNT

    std::vector<std::shared_ptr<Track>> unmatched_high_conf_detections;
    for (int detection_idx: unconfirmed_associations.unmatched_det_indices){
        const std::shared_ptr<Track> &detection = unmatched_detections_after_1st_association[detection_idx];
        unmatched_high_conf_detections.push_back(detection);
    }

    // Initialize new tracks for the high confidence detections left after all the associations
    for (const std::shared_ptr<Track> &detection:unmatched_high_conf_detections)
    {
        if (detection->get_score() >= _new_track_thresh)
        {
            detection->activate(*_kalman_filter, _frame_id);
            activated_tracks.push_back(detection);
            LOG_ASSIGNMENT(detection, detection, "")
        }
    }
    LOG_ITEM_END_LINE
    ////////////////// Initialize new tracks //////////////////


    ////////////////// Update lost tracks state //////////////////
    LOG_TITLE_STEP(_frame_id, "5", "Update state");
    LOG_TITLE_ARG("Killed tracks (track->end_frame() > max_time_lost=%d", (int)_max_time_lost)
    LOG_INIT_COUNT

    for (const std::shared_ptr<Track> &track: _lost_tracks)
    {
        if (_frame_id - track->end_frame() + 1 > _max_time_lost) // adding +1 to be consistent with the other methods. 
        {
            track->mark_removed();
            removed_tracks.push_back(track);
            LOG_TRACK(track)
        }
    }

    LOG_ITEM_END_LINE
    ////////////////// Update lost tracks state //////////////////


    ////////////////// Clean up the track lists //////////////////
    std::vector<std::shared_ptr<Track>> updated_tracked_tracks;
    for (const std::shared_ptr<Track> &_tracked_track: _tracked_tracks)
    {
        if (_tracked_track->state == TrackState::Tracked)
        {
            updated_tracked_tracks.push_back(_tracked_track);
        }
    }
    _tracked_tracks = _merge_track_lists(updated_tracked_tracks, activated_tracks);
    _tracked_tracks = _merge_track_lists(_tracked_tracks, refind_tracks);

    _lost_tracks = _merge_track_lists(_lost_tracks, lost_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, _tracked_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, removed_tracks);

    std::vector<std::shared_ptr<Track>> tracked_tracks_cleaned, lost_tracks_cleaned;

    LOG_TITLE_STEP(_frame_id, "6", "Remove duplicate tracks - IOU distance < 0.15 (fixed)")

    _remove_duplicate_tracks(tracked_tracks_cleaned, lost_tracks_cleaned, _tracked_tracks, _lost_tracks, _trace);
    _tracked_tracks = tracked_tracks_cleaned,
    _lost_tracks = lost_tracks_cleaned;

    // Note: as removed_tracks is a local shared pointers vector, it should clean itself when this function ends and avoid memory leak. 
     
    ////////////////// Clean up the track lists //////////////////


    ////////////////// Update output tracks //////////////////
    LOG_TITLE_STEP(_frame_id, "7", "Get final tracks output (U = unconfirmed/unactivated tracks) ")
    LOG_INIT_COUNT

    std::vector<std::shared_ptr<Track>> output_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks)
    {        
        if (track->is_activated){
            output_tracks.push_back(track);
            LOG_TRACK(track) 
        } else {
            output_tracks.push_back(track);
            LOG_TRACK_T(track, "U") // U for unconfirmed or unactivated track
        }
        LOG_ITEM_NEW_LINE(8)
    }
    ////////////////// Update output tracks //////////////////

    LOG_ITEM_END_LINE
    LOG_END

    return output_tracks;
}

FeatureVector 
BoTSORT::_get_features(double *featArray, int featDim) {

    FeatureVector feature_vector = FeatureVector::Zero(1, featDim);
    if (featArray == nullptr || featDim <= 0) {
        return feature_vector; // Return zero vector if input is invalid
    }
    for (int i = 0; i < featDim; i++) {
        feature_vector(0, i) = (float)featArray[i];
        // Eigen::Map<FeatureVector> feature_vector(featArray, 0, featDim); // can't do this as featArray is double* but feature_vector is float* for now. 
    }
    return feature_vector;
}

// FeatureVector BoTSORT::_extract_features(const cv::Mat &frame, const cv::Rect_<float> &bbox_tlwh)
// {
//     cv::Mat patch = frame(bbox_tlwh);
//     return _reid_model->extract_features(patch);
// }


std::vector<std::shared_ptr<Track>>
BoTSORT::_merge_track_lists(std::vector<std::shared_ptr<Track>> &tracks_list_a,
                            std::vector<std::shared_ptr<Track>> &tracks_list_b)
{
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> merged_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_list_a)
    {
        exists[track->track_id] = true;
        merged_tracks_list.push_back(track);
    }

    for (const std::shared_ptr<Track> &track: tracks_list_b)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            exists[track->track_id] = true;
            merged_tracks_list.push_back(track);
        }
    }

    return merged_tracks_list;
}


std::vector<std::shared_ptr<Track>> 
BoTSORT::_remove_from_list(std::vector<std::shared_ptr<Track>> &tracks_list,
                                     std::vector<std::shared_ptr<Track>> &tracks_to_remove)
{
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> new_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_to_remove)
    {
        exists[track->track_id] = true;
    }

    for (const std::shared_ptr<Track> &track: tracks_list)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            new_tracks_list.push_back(track);
        }
    }

    return new_tracks_list;
}


void BoTSORT::_remove_duplicate_tracks(
                                    std::vector<std::shared_ptr<Track>> &result_tracks_a,
                                    std::vector<std::shared_ptr<Track>> &result_tracks_b,
                                    std::vector<std::shared_ptr<Track>> &tracks_list_a,
                                    std::vector<std::shared_ptr<Track>> &tracks_list_b,
                                    bool trace)
{
    CostMatrix iou_dists = iou_distance(tracks_list_a, tracks_list_b);

    std::unordered_set<size_t> dup_a, dup_b;
    for (Eigen::Index i = 0; i < iou_dists.rows(); i++)
    {
        for (Eigen::Index j = 0; j < iou_dists.cols(); j++)
        {
            if (iou_dists(i, j) < 0.15)
            {
                int time_a = static_cast<int>(tracks_list_a[i]->frame_id -
                                              tracks_list_a[i]->start_frame);
                int time_b = static_cast<int>(tracks_list_b[j]->frame_id -
                                              tracks_list_b[j]->start_frame);

                // We make an assumption that the longer trajectory is the correct one
                if (time_a > time_b){
                    dup_b.insert(j);// In list b, track with index j is a duplicate
                    LOG_DUPLICATE(tracks_list_b[j], tracks_list_a[i])
                } else{
                    dup_a.insert(i);// In list a, track with index i is a duplicate
                    LOG_DUPLICATE(tracks_list_a[i], tracks_list_b[j])
                }
            }
        }
    }

    // Remove duplicates from the lists
    for (size_t i = 0; i < tracks_list_a.size(); i++)
    {
        if (dup_a.find(i) == dup_a.end())
        {
            result_tracks_a.push_back(tracks_list_a[i]);
        }
    }

    for (size_t i = 0; i < tracks_list_b.size(); i++)
    {
        if (dup_b.find(i) == dup_b.end())
        {
            result_tracks_b.push_back(tracks_list_b[i]);
        }
    }
}

void BoTSORT::_load_params_from_config(const TrackerParams &config)
{
    _reid_enabled = config.reid_enabled;
    _gmc_enabled = config.gmc_enabled;
    _track_high_thresh = config.track_high_thresh;
    _track_low_thresh = config.track_low_thresh;
    _new_track_thresh = config.new_track_thresh;
    _track_buffer = config.track_buffer;
    _match_thresh = config.match_thresh;
    _proximity_thresh = config.proximity_thresh;
    _appearance_thresh = config.appearance_thresh;
    _gmc_method_name = config.gmc_method_name;
    _frame_rate = config.frame_rate;
    _lambda = config.lambda;
}


}// namespace bot_sort