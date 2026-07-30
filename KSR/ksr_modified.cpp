#include "include/Kinetic_surface_reconstruction_3.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/PLY.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

#include <CGAL/mst_orient_normals.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/convex_hull_2.h>

#include <algorithm>
#include <cmath>
// #include <limits>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Vector_3 = typename Kernel::Vector_3;
using Segment_3 = typename Kernel::Segment_3;

using Point_set = CGAL::Point_set_3<Point_3>;
using Point_map = typename Point_set::Point_map;
using Normal_map = typename Point_set::Vector_map;

using KSR = CGAL::Kinetic_surface_reconstruction_3<Kernel, Point_set, Point_map,
                                                   Normal_map>;

int main(int argc, char **argv) {
  // Input and CLI args.
  std::string input_file, output_dir;
  auto print_usage = [&](const char *prog) {
    std::cout << "Usage: " << prog
              << " [-i|--input <input.ply>] [-o|--output <output_dir>]"
              << std::endl;
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (a == "-i" || a == "--input") {
      if (i + 1 < argc)
        input_file = argv[++i];
      else {
        std::cerr << "Error: missing argument for " << a << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
    } else if (a == "-o" || a == "--output") {
      if (i + 1 < argc)
        output_dir = argv[++i];
      else {
        std::cerr << "Error: missing argument for " << a << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
    } else {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  namespace fs = std::filesystem;
  if (!fs::exists(input_file)) {
    std::cerr << "Input file not found: " << input_file << std::endl;
    return EXIT_FAILURE;
  }

  fs::path outdir(output_dir);
  std::error_code ec;
  if (!fs::exists(outdir)) {
    if (!fs::create_directories(outdir, ec)) {
      std::cerr << "Failed to create output directory '" << output_dir
                << "': " << ec.message() << std::endl;
      return EXIT_FAILURE;
    }
  } else if (!fs::is_directory(outdir)) {
    std::cerr << "Output path exists and is not a directory: " << output_dir
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Reading input: " << input_file << std::endl;
  std::cout << "Writing outputs to: " << outdir << std::endl;

  Point_set point_set;
  auto assignment_prop =
      point_set.add_property_map<int>("pts_ins_assignment", 0).first;

  if (!CGAL::IO::read_point_set(input_file, point_set)) {
    std::cerr << "Failed to read point set from: " << input_file << std::endl;
    return EXIT_FAILURE;
  }

  bool need_normals = false;
  if (point_set.has_normal_map() && point_set.begin() != point_set.end()) {
    auto n = point_set.normal(*point_set.begin());
    if (n.squared_length() < 1e-6) {
      need_normals = true;
    }
  } else {
    need_normals = true;
  }

  if (need_normals) {
    std::cout
        << "Normals are missing or zero. Estimating and orienting normals..."
        << std::endl;
    if (!point_set.has_normal_map()) {
      point_set.add_normal_map();
    }
    CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
        point_set, 12,
        point_set.parameters()
            .point_map(point_set.point_map())
            .normal_map(point_set.normal_map()));
    CGAL::mst_orient_normals(point_set, 12,
                             point_set.parameters()
                                 .point_map(point_set.point_map())
                                 .normal_map(point_set.normal_map()));
  }

  bool has_assignment = false;
  std::vector<int> plane_assignments;
  plane_assignments.reserve(point_set.size());
  for (auto it = point_set.begin(); it != point_set.end(); ++it) {
    int val = assignment_prop[*it];
    plane_assignments.push_back(val);
    if (val > 0) {
      has_assignment = true;
    }
  }

  // ─── Ground-plane detection & bottom-cap synthesis ──────────────────────
  // For each labeled plane, we project its points into the plane's own local
  // (U,V) tangent frame, compute the 2D convex-hull area, and declare the
  // plane with the LARGEST area the "ground". A dense grid of synthetic
  // points is then appended to point_set with a new label so KSR can close
  // the bottom of the building shell.
  //
  // Set to false to skip ground-cap synthesis entirely.
  const bool kEnableGroundCap = true;
  if (kEnableGroundCap && has_assignment) {
    const double kGridStep = 0.05;

    // ── Helper: orthonormal tangent frame (U,V) for a unit normal N ────────
    auto make_tangent_frame = [](Vector_3 N, Vector_3 &U, Vector_3 &V) {
      // Pick an arbitrary vector not (nearly) parallel to N.
      Vector_3 tmp = (std::abs(CGAL::to_double(N.z())) < 0.9)
                         ? Vector_3(0.0, 0.0, 1.0)
                         : Vector_3(1.0, 0.0, 0.0);
      U = CGAL::cross_product(tmp, N);
      double lu = std::sqrt(CGAL::to_double(U.squared_length()));
      if (lu > 1e-12) U = U / lu;
      V = CGAL::cross_product(N, U);
      double lv = std::sqrt(CGAL::to_double(V.squared_length()));
      if (lv > 1e-12) V = V / lv;
    };

    // ── Helper: signed polygon area via shoelace (Kernel::Point_2) ─────────
    using Point_2 = typename Kernel::Point_2;
    auto poly_area_2d = [](const std::vector<Point_2> &hull) -> double {
      double area = 0.0;
      int n = static_cast<int>(hull.size());
      for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += CGAL::to_double(hull[i].x()) * CGAL::to_double(hull[j].y());
        area -= CGAL::to_double(hull[j].x()) * CGAL::to_double(hull[i].y());
      }
      return std::abs(area) * 0.5;
    };

    // ── Pass 1: accumulate centroid and average normal per plane ────────────
    int max_lbl = *std::max_element(plane_assignments.begin(),
                                    plane_assignments.end());

    struct PlaneAccum {
      double cx = 0, cy = 0, cz = 0; // centroid sum
      double nx = 0, ny = 0, nz = 0; // normal sum
      int count = 0;
    };
    std::vector<PlaneAccum> accum(max_lbl + 1);

    {
      int idx = 0;
      for (auto it = point_set.begin(); it != point_set.end(); ++it, ++idx) {
        int lbl = plane_assignments[idx];
        if (lbl <= 0) continue;
        const auto &p = point_set.point(*it);
        const auto &n = point_set.normal(*it);
        auto &a = accum[lbl];
        a.cx += CGAL::to_double(p.x());
        a.cy += CGAL::to_double(p.y());
        a.cz += CGAL::to_double(p.z());
        a.nx += CGAL::to_double(n.x());
        a.ny += CGAL::to_double(n.y());
        a.nz += CGAL::to_double(n.z());
        ++a.count;
      }
    }

    // ── Precompute (N, U, V, C) frame per plane ─────────────────────────────
    struct PlaneFrame {
      Point_3 C;
      Vector_3 N, U, V;
      bool valid = false;
    };
    std::vector<PlaneFrame> frames(max_lbl + 1);
    for (int lbl = 1; lbl <= max_lbl; ++lbl) {
      auto &a = accum[lbl];
      if (a.count == 0) continue;
      double nlen = std::sqrt(a.nx * a.nx + a.ny * a.ny + a.nz * a.nz);
      if (nlen < 1e-12) continue;
      auto &f = frames[lbl];
      f.N = Vector_3(a.nx / nlen, a.ny / nlen, a.nz / nlen);
      make_tangent_frame(f.N, f.U, f.V);
      f.C = Point_3(a.cx / a.count, a.cy / a.count, a.cz / a.count);
      f.valid = true;
    }

    // ── Pass 2: project points into each plane's (U,V) frame ───────────────
    std::vector<std::vector<Point_2>> pts2d(max_lbl + 1);
    struct UVBBox {
      double umin = 1e18, umax = -1e18;
      double vmin = 1e18, vmax = -1e18;
    };
    std::vector<UVBBox> uvbb(max_lbl + 1);

    {
      int idx = 0;
      for (auto it = point_set.begin(); it != point_set.end(); ++it, ++idx) {
        int lbl = plane_assignments[idx];
        if (lbl <= 0 || !frames[lbl].valid) continue;
        const auto &f = frames[lbl];
        const auto &p = point_set.point(*it);
        Vector_3 dp = p - f.C;
        double u = CGAL::to_double(dp * f.U);
        double v = CGAL::to_double(dp * f.V);
        pts2d[lbl].emplace_back(u, v);
        auto &bb = uvbb[lbl];
        bb.umin = std::min(bb.umin, u); bb.umax = std::max(bb.umax, u);
        bb.vmin = std::min(bb.vmin, v); bb.vmax = std::max(bb.vmax, v);
      }
    }

    // ── Compute convex-hull area per plane; pick ground (max area) ──────────
    int ground_lbl = -1;
    double best_area = -1.0;
    std::cout << "Ground-plane detection (convex-hull areas):" << std::endl;
    for (int lbl = 1; lbl <= max_lbl; ++lbl) {
      if (static_cast<int>(pts2d[lbl].size()) < 3) continue;
      std::vector<Point_2> hull;
      CGAL::convex_hull_2(pts2d[lbl].begin(), pts2d[lbl].end(),
                          std::back_inserter(hull));
      double area = poly_area_2d(hull);
      std::cout << "  Plane " << lbl << ": ch-area=" << area << std::endl;
      if (area > best_area) { best_area = area; ground_lbl = lbl; }
    }

    // ── Synthesize grid on the ground plane ─────────────────────────────────
    if (ground_lbl > 0) {
      const auto &f  = frames[ground_lbl];
      const auto &bb = uvbb[ground_lbl];
      int new_lbl = ground_lbl; // reuse the same label so both planes merge

      std::cout << "Ground plane: label=" << ground_lbl
                << "  centroid=(" << CGAL::to_double(f.C.x())
                << "," << CGAL::to_double(f.C.y())
                << "," << CGAL::to_double(f.C.z()) << ")"
                << "  ch-area=" << best_area
                << "  (synthetic points will share label=" << new_lbl << ")" << std::endl;

      if (!point_set.has_normal_map()) point_set.add_normal_map();

      int added = 0;
      for (double u = bb.umin; u <= bb.umax + kGridStep * 0.5; u += kGridStep) {
        for (double v = bb.vmin; v <= bb.vmax + kGridStep * 0.5; v += kGridStep) {
          Point_3 P = f.C + u * f.U + v * f.V;
          auto it_new = point_set.insert(P);
          point_set.normal(*it_new) = f.N;
          plane_assignments.push_back(new_lbl);
          ++added;
        }
      }
      std::cout << "  Injected " << added
                << " synthetic ground-cap points (label=" << new_lbl << ")"
                << std::endl;

      // Write the augmented point cloud for visual inspection.
      fs::path cap_ply = outdir / "gnd_capped.ply";
      if (CGAL::IO::write_point_set(cap_ply.string(), point_set))
        std::cout << "  Wrote capped point set: " << cap_ply << std::endl;
      else
        std::cerr << "  Failed to write capped point set: " << cap_ply << std::endl;
    } else {
      std::cout << "Ground plane detection: no plane with sufficient points, "
                   "skipping bottom-cap synthesis."
                << std::endl;
    }
  }
  // ─── End ground-plane detection ──────────────────────────────────────────

  std::map<typename KSR::KSP::Face_support, bool> external_nodes;
  // All bbox faces prefer "outside" label except YMAX (intentional for model orientation).
  external_nodes[KSR::KSP::Face_support::ZMIN] = true;
  // external_nodes[KSR::KSP::Face_support::ZMIN] = false;
  external_nodes[KSR::KSP::Face_support::ZMAX] = false;
  external_nodes[KSR::KSP::Face_support::XMIN] = false;
  external_nodes[KSR::KSP::Face_support::XMAX] = false;
  external_nodes[KSR::KSP::Face_support::YMIN] = false;
  // external_nodes[KSR::KSP::Face_support::YMAX] = true;
  external_nodes[KSR::KSP::Face_support::YMAX] = false;
 
  auto param =CGAL::parameters::k_neighbors(8)
    .maximum_distance(0.05) // the maximum distance from a point to a plane
    .maximum_angle(15)  // the maximum angle in degrees between the normal associated
                        // with a point and the normal of a plane
    .minimum_region_size(400) // the minimum number of points a region must have
    .regularize_parallelism(true) // whether parallelism should be regularized or not
    .regularize_coplanarity(true) // whether coplanarity should be regularized or not
    .regularize_orthogonality(true) // whether orthogonality should be regularized or not
    .angle_tolerance(15)   // Idk
    .maximum_offset(0.05); // maximum distance between two parallel planes
                            // to be considered coplanar


  // Algorithm.
  KSR ksr(point_set, param);
  if (has_assignment) {
    std::cout
        << "Found pts_ins_assignment property. Using external plane detections."
        << std::endl;
    std::cout << "  Point set size: " << point_set.size() << std::endl;
    std::cout << "  Plane assignments size: " << plane_assignments.size()
              << std::endl;
    int max_label = 0;
    for (int v : plane_assignments)
      max_label = std::max(max_label, v);
    std::cout << "  Max plane label: " << max_label << std::endl;
    ksr.injection_and_partition(plane_assignments, 2, param);
    std::cout << "injection_and_partition completed." << std::endl;
    std::cout << "  Number of volumes: "
              << ksr.kinetic_partition().number_of_volumes() << std::endl;
    // Diagnostic: how many planes did KSP actually receive?
    // If less than max_label, coplanar planes were deduplicated by KSP.
    // If output contains axis-aligned phantom planes, octree subdivision is the cause.
    std::cout << "  KSP input_planes count: "
              << ksr.kinetic_partition().input_planes().size()
              << " (injected " << max_label << " labeled planes)" << std::endl;
  } else {
    std::cout << "No pts_ins_assignment property found. Falling back to "
                 "internal CGAL shape detection."
              << std::endl;
    ksr.detection_and_partition(2, param);
  }

  std::vector<Point_3> vtx;
  std::vector<std::vector<std::size_t>> polylist;
  std::vector<FT> lambdas{0.1, 0.3, 0.5, 0.7, 0.9};

  bool save_biggest_component_only = true;
  bool non_empty = false;
  for (FT l : lambdas) {
    vtx.clear();
    polylist.clear();
    // std::cout << "Reconstructing with lambda=" << CGAL::to_double(l) << "..." << std::endl;
    ksr.reconstruct_with_ground(l, std::back_inserter(vtx), std::back_inserter(polylist));
    // ksr.reconstruct(l, external_nodes, std::back_inserter(vtx), std::back_inserter(polylist));
    
    std::cout << "  => vtx=" << vtx.size() << " polylist=" << polylist.size() << std::endl;

    if (polylist.size() > 0) {
      non_empty = true;

      // Repair the soup: removes duplicates and degenerated faces
      CGAL::Polygon_mesh_processing::repair_polygon_soup(vtx, polylist);
      // Orient the soup: fixes inconsistent normals which cause non-manifold errors
      CGAL::Polygon_mesh_processing::orient_polygon_soup(vtx, polylist);

      using Mesh = CGAL::Surface_mesh<Point_3>;

      std::string lstr = std::to_string(CGAL::to_double(l));
      std::string filename = "polylist_" + lstr + ".ply";
      fs::path outp = outdir / filename;

      if (save_biggest_component_only) {
        // Convert polygon soup to a Surface_mesh
        Mesh full_mesh;
        CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(vtx, polylist, full_mesh);

        // Label connected components: each face gets an integer component ID
        Mesh::Property_map<Mesh::Face_index, std::size_t> comp_id =
          full_mesh.add_property_map<Mesh::Face_index, std::size_t>("f:component", 0).first;
        std::size_t num_components = CGAL::Polygon_mesh_processing::connected_components(full_mesh, comp_id);

        std::cout << "Lambda " << CGAL::to_double(l)
                  << ": " << num_components << " component(s)" << std::endl;

        // Count faces per component, find the largest
        std::vector<std::size_t> face_count(num_components, 0);
        for (Mesh::Face_index f : full_mesh.faces())
          ++face_count[comp_id[f]];

        std::size_t best = std::max_element(face_count.begin(), face_count.end()) - face_count.begin();
        std::cout << "  Largest component: " << best
                  << " (" << face_count[best] << " faces)" << std::endl;

        // Build sub-mesh for the largest component only
        Mesh sub_mesh;
        std::unordered_map<Mesh::Vertex_index, Mesh::Vertex_index> vmap;

        for (Mesh::Face_index f : full_mesh.faces()) {
          if (comp_id[f] != best) continue;

          std::vector<Mesh::Vertex_index> new_verts;
          for (Mesh::Vertex_index v :
               CGAL::vertices_around_face(full_mesh.halfedge(f), full_mesh)) {
            auto it = vmap.find(v);
            if (it == vmap.end()) {
              Mesh::Vertex_index nv = sub_mesh.add_vertex(full_mesh.point(v));
              vmap[v] = nv;
              new_verts.push_back(nv);
            } else {
              new_verts.push_back(it->second);
            }
          }
          sub_mesh.add_face(new_verts);
        }

        bool success = CGAL::IO::write_PLY(outp.string(), sub_mesh);
        if (success)
          std::cout << "  Wrote largest component: " << outp << std::endl;
        else
          std::cout << "  Failed to write: " << outp << std::endl;

      } else {
        // Save the full mesh (all components) as a polygon soup
        std::cout << "Lambda " << CGAL::to_double(l) << ": saving full soup" << std::endl;
        bool success = CGAL::IO::write_polygon_soup(outp.string(), vtx, polylist);
        if (success)
          std::cout << "  Wrote: " << outp << std::endl;
        else
          std::cout << "  Failed to write: " << outp << std::endl;
      }
    }
  }

  return (non_empty) ? EXIT_SUCCESS : EXIT_FAILURE;
}